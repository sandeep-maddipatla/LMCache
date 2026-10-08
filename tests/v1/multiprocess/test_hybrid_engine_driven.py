# SPDX-License-Identifier: Apache-2.0
"""Tests for hybrid (multi engine-group) KV transfer on the engine-driven path."""

# Standard
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock
import logging

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context import (
    EngineDrivenTransferContext,
    hybrid_engine_driven,
    worker_transfer,
)
from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
    AsyncEngineDrivenTransferContext,
)
from lmcache.v1.multiprocess.transfer_context.hybrid_engine_driven import (
    HybridLayout,
    build_hybrid_layout,
    gather_hybrid_paged_kv_to_cpu,
    null_chunk_indices,
    scatter_hybrid_cpu_to_paged_kv,
)

NUM_LAYERS, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE = 4, 8, 4, 2, 8

# Layers 0, 2 in engine group 0 and 1, 3 in engine group 1, like the
# interleaved linear-attention / full-attention layers of Qwen3.5-style models.
TWO_GROUPS = [
    EngineGroupInfo(engine_group_id=0, layer_indices=(0, 2)),
    EngineGroupInfo(engine_group_id=1, layer_indices=(1, 3)),
]
# Engine group 0 split into two LMCache groups (different list positions).
SPLIT_GROUP = [
    EngineGroupInfo(engine_group_id=0, layer_indices=(0,)),
    EngineGroupInfo(engine_group_id=1, layer_indices=(1, 3)),
    EngineGroupInfo(engine_group_id=0, layer_indices=(2,)),
]
LAYER_GROUP = {0: 0, 1: 1, 2: 0, 3: 1}
# Layers given a different dtype and width: all of engine group 1, or only
# layer 2, so the two LMCache groups of engine group 0 differ.
GROUP1_MIXED = (1, 3)
SPLIT_MIXED = (2,)


@pytest.fixture(autouse=True)
def _host_independent_kv_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep KV format discovery independent of the host running the test."""
    monkeypatch.setattr(
        "lmcache.v1.gpu_connector.kv_format.detectors.vllm.torch_device_type",
        torch_device_type if torch_device_type != "cpu" else "cuda",
    )


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def record_warnings() -> Iterator[Callable[[Any], _RecordingHandler]]:
    """Record WARNING+ logs of a module's ``logger``.

    lmcache loggers do not propagate, so caplog cannot see them.
    """
    attached: list[tuple[logging.Logger, _RecordingHandler]] = []

    def _attach(module: Any) -> _RecordingHandler:
        handler = _RecordingHandler()
        module.logger.addHandler(handler)
        attached.append((module.logger, handler))
        return handler

    yield _attach
    for module_logger, handler in attached:
        module_logger.removeHandler(handler)


def _kv_caches(
    fill: bool = True,
    mixed: tuple[int, ...] = (),
    group1_block_size: int = BLOCK_SIZE,
) -> dict[str, torch.Tensor]:
    """Per-layer NHD tensors ``[2, num_blocks, block_size, heads, head]``.

    Layers in ``mixed`` use ``bfloat16`` and one head, so their token width
    (16 bytes) differs from the others' (64 bytes), like the int8 Mamba state
    view next to bf16 attention on vLLM >= 0.26.
    """
    make = torch.randn if fill else torch.zeros
    caches = {}
    for i in range(NUM_LAYERS):
        heads, dtype = (1, torch.bfloat16) if i in mixed else (NUM_HEADS, None)
        block_size = group1_block_size if LAYER_GROUP[i] == 1 else BLOCK_SIZE
        caches[f"layer_{i}"] = make(
            2,
            NUM_BLOCKS,
            block_size,
            heads,
            HEAD_SIZE,
            dtype=dtype,
            device=torch_device_type,
        )
    return caches


def _block(kv: torch.Tensor, block: int) -> torch.Tensor:
    """One paged block as ``[2, block_size, heads * head]`` on the host."""
    return kv[:, block].reshape(2, kv.shape[2], -1).cpu()


def _layout(infos: list[EngineGroupInfo], mixed: tuple[int, ...] = ()) -> HybridLayout:
    layout = build_hybrid_layout(infos, _kv_caches(mixed=mixed))
    assert layout is not None
    return layout


def _group_layers(layout: HybridLayout | None) -> list[tuple[int, list[int]]]:
    assert layout is not None
    return [(group.engine_group_id, group.layer_indices) for group in layout.groups]


def _key(tokens: int) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "m", 1, 0, [1] * tokens, start=0, end=tokens, request_id="req"
    )


def test_build_enables_exact_hybrid_layout() -> None:
    layout = build_hybrid_layout(TWO_GROUPS, _kv_caches())
    assert _group_layers(layout) == [(0, [0, 2]), (1, [1, 3])]
    assert layout is not None
    # Uniform groups keep the engine dtype and width, like a single group.
    assert not layout.byte_objects
    assert layout.object_dtype == torch.float32
    assert layout.object_hidden_dim == NUM_HEADS * HEAD_SIZE


def test_build_uses_byte_objects_for_mixed_dtype_and_width() -> None:
    layout = build_hybrid_layout(TWO_GROUPS, _kv_caches(mixed=GROUP1_MIXED))
    assert _group_layers(layout) == [(0, [0, 2]), (1, [1, 3])]
    assert layout is not None
    assert layout.byte_objects
    assert layout.object_dtype == torch.uint8
    # Widest group: float32 with NUM_HEADS heads.
    assert layout.object_hidden_dim == NUM_HEADS * HEAD_SIZE * 4


def test_build_enables_align_mode_mamba_one_block_window() -> None:
    """Align-mode Mamba groups report a one-block window (sw == block size)."""
    infos = [
        EngineGroupInfo(engine_group_id=0, layer_indices=(0, 2), tokens_per_block=16),
        EngineGroupInfo(
            engine_group_id=1,
            layer_indices=(1, 3),
            tokens_per_block=16,
            sw_size_tokens=16,
        ),
    ]
    assert _group_layers(build_hybrid_layout(infos, _kv_caches())) == [
        (0, [0, 2]),
        (1, [1, 3]),
    ]


def test_build_moves_each_lmcache_group_of_one_engine_group() -> None:
    """Block ids are indexed by engine group, not by LMCache group position."""
    layout = build_hybrid_layout(SPLIT_GROUP, _kv_caches())
    assert _group_layers(layout) == [(0, [0]), (0, [2]), (1, [1, 3])]
    assert layout is not None
    assert layout.num_engine_groups == 2


@pytest.mark.parametrize(
    "infos",
    [
        pytest.param(
            [EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1, 2, 3))],
            id="single_group",
        ),
        pytest.param(
            [
                EngineGroupInfo(engine_group_id=0, layer_indices=(0, 2)),
                EngineGroupInfo(engine_group_id=0, layer_indices=(1, 3)),
            ],
            id="one_engine_group_split",
        ),
        pytest.param(
            [
                EngineGroupInfo(engine_group_id=0, layer_indices=(0,)),
                EngineGroupInfo(engine_group_id=1, layer_indices=(1, 3)),
            ],
            id="layer_in_no_group",
        ),
        pytest.param(
            [
                EngineGroupInfo(
                    engine_group_id=0, layer_indices=(0, 2), tokens_per_block=16
                ),
                EngineGroupInfo(
                    engine_group_id=1, layer_indices=(1, 3), tokens_per_block=32
                ),
            ],
            id="mixed_tokens_per_block",
        ),
        pytest.param(
            [
                EngineGroupInfo(engine_group_id=0, layer_indices=(0, 2)),
                EngineGroupInfo(
                    engine_group_id=1, layer_indices=(1, 3), sw_size_tokens=128
                ),
            ],
            id="sliding_window",
        ),
        pytest.param(
            [
                EngineGroupInfo(engine_group_id=0, layer_indices=(0, 2)),
                EngineGroupInfo(engine_group_id=2, layer_indices=(1, 3)),
            ],
            id="non_dense_ids",
        ),
    ],
)
def test_build_disables_unsupported_layouts(infos: list[EngineGroupInfo]) -> None:
    assert build_hybrid_layout(infos, _kv_caches()) is None


def test_build_ignores_one_engine_group_without_warning(
    record_warnings: Callable[[Any], _RecordingHandler],
) -> None:
    """A non-hybrid model split into LMCache groups is not a disabled hybrid."""
    warnings = record_warnings(hybrid_engine_driven)
    infos = [
        EngineGroupInfo(engine_group_id=0, layer_indices=(0, 2)),
        EngineGroupInfo(engine_group_id=0, layer_indices=(1, 3)),
    ]
    assert build_hybrid_layout(infos, _kv_caches()) is None
    assert warnings.records == []


def test_build_disables_groups_with_different_block_sizes() -> None:
    """A chunk must span the same blocks and tokens in every group."""
    caches = _kv_caches(group1_block_size=2 * BLOCK_SIZE)
    assert build_hybrid_layout(TWO_GROUPS, caches) is None


def test_null_chunk_indices_flags_chunks_null_in_any_group() -> None:
    layout = _layout(TWO_GROUPS)
    assert null_chunk_indices([[3, 5, 6], [0, 7, 4]], layout, 1, [0, 1, 2]) == [0]
    assert null_chunk_indices([[3, 5, 6], [0, 7, 4]], layout, 1, [1, 2]) == []


@pytest.mark.parametrize(
    "block_ids", [[[1], [2], [3]], [[1, 2], [3]]], ids=["group_count", "block_count"]
)
def test_null_chunk_indices_rejects_mismatched_block_ids(
    block_ids: list[list[int]],
) -> None:
    layout = _layout(TWO_GROUPS)
    with pytest.raises(ValueError):
        null_chunk_indices(block_ids, layout, 1, [0])


@pytest.mark.parametrize(
    ("infos", "mixed"),
    [
        pytest.param(TWO_GROUPS, (), id="two_groups"),
        pytest.param(SPLIT_GROUP, (), id="split_group"),
        pytest.param(TWO_GROUPS, GROUP1_MIXED, id="two_groups_byte_objects"),
        pytest.param(SPLIT_GROUP, SPLIT_MIXED, id="split_group_byte_objects"),
    ],
)
def test_gather_scatter_round_trip_uses_each_groups_blocks(
    infos: list[EngineGroupInfo], mixed: tuple[int, ...]
) -> None:
    layout = _layout(infos, mixed=mixed)
    source = _kv_caches(mixed=mixed)
    chunks = gather_hybrid_paged_kv_to_cpu(source, [[3, 5], [4, 7]], layout, 1)
    assert [(tuple(c.shape), c.dtype) for c in chunks] == [
        (
            (2, NUM_LAYERS, BLOCK_SIZE, layout.object_hidden_dim),
            layout.object_dtype,
        )
    ] * 2

    destination = _kv_caches(fill=False, mixed=mixed)
    # Chunk 0 is null in both groups (align-mode Mamba keeps one state block).
    scatter_hybrid_cpu_to_paged_kv(destination, [[0, 6], [0, 2]], layout, chunks, 1)
    torch_dev.synchronize()
    for layer in range(NUM_LAYERS):
        name = f"layer_{layer}"
        src_block, dst_block, other = (
            (5, 6, 2) if LAYER_GROUP[layer] == 0 else (7, 2, 6)
        )
        assert torch.equal(
            _block(destination[name], dst_block), _block(source[name], src_block)
        )
        assert torch.count_nonzero(destination[name][:, 0]) == 0, "null block written"
        assert torch.count_nonzero(destination[name][:, other]) == 0, (
            "wrong group's block"
        )


def test_gather_fills_out_and_ignores_extra_buffers() -> None:
    layout = _layout(TWO_GROUPS)
    source = _kv_caches()
    expected = gather_hybrid_paged_kv_to_cpu(source, [[3, 5], [4, 7]], layout, 1)
    out = [torch.zeros_like(expected[0]) for _ in range(3)]
    chunks = gather_hybrid_paged_kv_to_cpu(source, [[3, 5], [4, 7]], layout, 1, out=out)
    assert len(chunks) == 2
    for chunk, buffer, want in zip(chunks, out, expected, strict=False):
        assert chunk is buffer
        assert torch.equal(chunk, want)
    assert torch.count_nonzero(out[2]) == 0, "extra buffer written"


def test_gather_rejects_too_few_out_buffers() -> None:
    layout = _layout(TWO_GROUPS)
    source = _kv_caches()
    out = [torch.empty(2, NUM_LAYERS, BLOCK_SIZE, layout.object_hidden_dim)]
    with pytest.raises(ValueError):
        gather_hybrid_paged_kv_to_cpu(source, [[3, 5], [4, 7]], layout, 1, out=out)


def test_scatter_skips_null_blocks_and_skipped_prefix() -> None:
    """Two blocks per chunk, a one-block skip, and a null chunk in group 1."""
    layout = _layout(TWO_GROUPS)
    blocks_per_chunk = 2
    chunk_tokens = blocks_per_chunk * BLOCK_SIZE
    chunks = [
        torch.randn(2, NUM_LAYERS, chunk_tokens, NUM_HEADS * HEAD_SIZE)
        for _ in range(3)
    ]
    block_ids = [[1, 2, 3, 4, 5, 6], [7, 6, 0, 0, 5, 4]]
    destination = _kv_caches(fill=False)
    scatter_hybrid_cpu_to_paged_kv(
        destination,
        block_ids,
        layout,
        chunks,
        blocks_per_chunk,
        skip_first_n_tokens=BLOCK_SIZE,
    )
    torch_dev.synchronize()
    for layer in range(NUM_LAYERS):
        name, gid = f"layer_{layer}", LAYER_GROUP[layer]
        assert torch.count_nonzero(destination[name][:, 0]) == 0, "null block written"
        for chunk_idx in range(3):
            for offset in range(blocks_per_chunk):
                block = block_ids[gid][chunk_idx * blocks_per_chunk + offset]
                if block == 0:
                    continue
                got = _block(destination[name], block)
                if chunk_idx == 0 and offset == 0:
                    assert torch.count_nonzero(got) == 0, "skipped block written"
                else:
                    want = chunks[chunk_idx][
                        :, layer, offset * BLOCK_SIZE : (offset + 1) * BLOCK_SIZE
                    ]
                    assert torch.equal(got, want.to(got.dtype))


class _FakeEngineDrivenContext:
    """Server stand-in: keeps committed chunks and serves them back."""

    def __init__(self, layout_desc: MemoryLayoutDesc) -> None:
        self.committed: list[torch.Tensor] | None = None
        self.commit_count = 0
        self.layout_desc = layout_desc

    def prepare_store(self, *_args: Any) -> None:
        return None

    def commit_store(
        self, _key: Any, _instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        for chunk in chunks:
            assert chunk.shape == self.layout_desc.shapes[0]
            assert chunk.dtype == self.layout_desc.dtypes[0]
        self.commit_count += 1
        self.committed = [c.clone() for c in chunks]
        return True

    def prepare_retrieve(self, *_args: Any) -> list[torch.Tensor] | None:
        return self.committed

    def commit_retrieve(self, *_args: Any) -> bool:
        return True

    def close(self) -> None:
        return None


def _registered(
    monkeypatch: pytest.MonkeyPatch,
    ctx_cls: type[EngineDrivenTransferContext],
    infos: list[EngineGroupInfo],
    mixed: tuple[int, ...] = (),
) -> tuple[EngineDrivenTransferContext, _FakeEngineDrivenContext, MagicMock]:
    servers: list[_FakeEngineDrivenContext] = []

    def _create(metadata: Any, *_a: Any, **_k: Any) -> _FakeEngineDrivenContext:
        servers.append(_FakeEngineDrivenContext(metadata.layout_desc))
        return servers[0]

    monkeypatch.setattr(worker_transfer, "create_engine_driven_context", _create)
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.return_value = future
    ctx = ctx_cls(1, req_client)
    ctx.register(
        kv_caches=_kv_caches(mixed=mixed),
        model_name="m",
        world_size=1,
        blocks_in_chunk=1,
        mq_timeout=1.0,
        engine_group_infos=infos,
    )
    return ctx, servers[0], req_client


def _registered_payload(req_client: MagicMock) -> Any:
    return req_client.register_kv_cache_engine_driven_context.call_args.args[0]


@pytest.mark.parametrize(
    "ctx_cls", [EngineDrivenTransferContext, AsyncEngineDrivenTransferContext]
)
@pytest.mark.parametrize("mixed", [(), SPLIT_MIXED], ids=["uniform", "byte_objects"])
def test_transfer_context_round_trips_hybrid_store_and_retrieve(
    monkeypatch: pytest.MonkeyPatch,
    ctx_cls: type[EngineDrivenTransferContext],
    mixed: tuple[int, ...],
) -> None:
    ctx, server, req_client = _registered(
        monkeypatch, ctx_cls, SPLIT_GROUP, mixed=mixed
    )
    payload = _registered_payload(req_client)
    if mixed:
        assert (payload.dtype_str, payload.hidden_dim_size) == (
            "uint8",
            NUM_HEADS * HEAD_SIZE * 4,
        )
    else:
        assert (payload.dtype_str, payload.hidden_dim_size) == (
            "float32",
            NUM_HEADS * HEAD_SIZE,
        )
    source = _kv_caches(mixed=mixed)
    event = ctx.create_recorded_event()
    stored = ctx.submit_store(
        "req", _key(2 * BLOCK_SIZE), source, [[3, 5], [4, 7]], event, 1
    )
    assert stored.result(timeout=10) is True
    assert server.commit_count == 1

    destination = _kv_caches(fill=False, mixed=mixed)
    retrieved = ctx.submit_retrieve(
        "req", _key(2 * BLOCK_SIZE), destination, [[1, 6], [5, 2]], event, 1
    )
    assert retrieved.result(timeout=10) is True
    for layer in range(NUM_LAYERS):
        name = f"layer_{layer}"
        for src_block, dst_block in (
            [(3, 1), (5, 6)] if LAYER_GROUP[layer] == 0 else [(4, 5), (7, 2)]
        ):
            assert torch.equal(
                _block(destination[name], dst_block), _block(source[name], src_block)
            )
    ctx.close()


@pytest.mark.parametrize(
    "ctx_cls", [EngineDrivenTransferContext, AsyncEngineDrivenTransferContext]
)
def test_transfer_context_skips_store_with_null_chunk(
    monkeypatch: pytest.MonkeyPatch,
    ctx_cls: type[EngineDrivenTransferContext],
    record_warnings: Callable[[Any], _RecordingHandler],
) -> None:
    ctx, server, _ = _registered(monkeypatch, ctx_cls, TWO_GROUPS)
    warnings = record_warnings(worker_transfer)
    for _ in range(2):
        stored = ctx.submit_store(
            "req",
            _key(2 * BLOCK_SIZE),
            _kv_caches(),
            [[3, 5], [0, 7]],
            ctx.create_recorded_event(),
            1,
        )
        assert stored.result(timeout=10) is False
    assert server.commit_count == 0
    # Skips are silent data loss for the cache, so the first one is a warning.
    assert ["Skipping hybrid store" in r.getMessage() for r in warnings.records] == [
        True
    ]
    ctx.close()


def test_transfer_context_keeps_rejecting_unsupported_multi_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sliding = [
        EngineGroupInfo(engine_group_id=0, layer_indices=(0, 2)),
        EngineGroupInfo(engine_group_id=1, layer_indices=(1, 3), sw_size_tokens=128),
    ]
    ctx, _, req_client = _registered(monkeypatch, EngineDrivenTransferContext, sliding)
    # Registration is unchanged when hybrid transfer is disabled.
    payload = _registered_payload(req_client)
    assert (payload.dtype_str, payload.hidden_dim_size) == (
        "float32",
        NUM_HEADS * HEAD_SIZE,
    )
    with pytest.raises(RuntimeError, match="does not support hybrid KV cache groups"):
        ctx.submit_store("req", _key(BLOCK_SIZE), _kv_caches(), [[3], [4]], None, 1)
    ctx.close()
