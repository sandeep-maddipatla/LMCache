# SPDX-License-Identifier: Apache-2.0
"""Hybrid (multi engine-group) KV transfer for the engine-driven path.

The engine-driven path stores one object per chunk with the layout
``[2, num_layers, chunk_tokens, hidden]`` (or ``[num_layers, ...]`` for
single-plane formats). Hybrid models such as Qwen3.5 / Qwen3.6 / Qwen3.8
(Gated-DeltaNet linear-attention + full-attention layers) keep each engine
group in its own paged block address space, so each engine group's layers are
gathered and scattered with that group's block ids.

After the KV cache group edits every layer exposes the same tokens per block,
but not always the same per-token width or dtype: on vLLM >= 0.26 the Mamba
state is an ``int8`` byte view while attention stays ``bfloat16``. When all
groups agree, objects keep the engine dtype, exactly as for a single-group
model. Otherwise the object is ``uint8`` and each token row holds a layer's
raw bytes, zero-padded to the widest group, which keeps a single object per
chunk without a server-side format change.

Align-mode Mamba groups point every chunk except the one holding the latest
recurrent state at the null block. Such chunks carry no valid KV, so a store
that contains one is skipped (object keys are content hashes; committing it
would serve garbage to a later prefix hit) and null blocks are never written
on retrieve.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.group_view import EngineGroupInfo, num_engine_groups
from lmcache.v1.multiprocess.transfer_context.base import (
    compute_kv_layout,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)

if TYPE_CHECKING:
    # First Party
    import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

# Engine block id that denotes absent KV data (vLLM's null block).
NULL_BLOCK_ID = 0


def _unsupported_group_reasons(
    engine_group_infos: Sequence[EngineGroupInfo], num_layers: int
) -> list[str]:
    """Return why the groups' metadata rules out hybrid transfer (empty if none)."""
    merged = _merged_layer_indices(engine_group_infos)
    reasons: list[str] = []
    if sorted(merged) != list(range(len(merged))):
        reasons.append(f"engine group ids not dense: {sorted(merged)}")
    if sorted(i for layers in merged.values() for i in layers) != list(
        range(num_layers)
    ):
        reasons.append(f"groups do not cover each of the {num_layers} layers once")
    tokens_per_block = {info.tokens_per_block for info in engine_group_infos}
    if len(tokens_per_block) != 1:
        reasons.append(f"mixed tokens_per_block {sorted(tokens_per_block)}")
    if any(
        info.sw_size_tokens not in (-1, info.tokens_per_block)
        for info in engine_group_infos
    ):
        reasons.append("sliding-window attention group present")
    return reasons


def _merged_layer_indices(
    engine_group_infos: Sequence[EngineGroupInfo],
) -> dict[int, list[int]]:
    """Map each engine group id to the layers of all its LMCache groups."""
    merged: dict[int, list[int]] = {}
    for info in engine_group_infos:
        merged.setdefault(info.engine_group_id, []).extend(info.layer_indices)
    return merged


def _group_layouts(
    engine_group_infos: Sequence[EngineGroupInfo],
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None,
) -> tuple[list[HybridGroupLayout], list[str]]:
    """Detect each LMCache group's layout; also return why it is unsupported.

    LMCache groups split an engine group by physical transfer identity (e.g.
    differing hidden dims), so each is detected and moved on its own.

    Raises:
        ValueError: If a group's KV format cannot be detected.
    """
    names = list(kv_caches)
    groups: list[HybridGroupLayout] = []
    reasons: list[str] = []
    geometry: set[tuple[int, int]] = set()
    for gid, layers in sorted(
        (info.engine_group_id, sorted(info.layer_indices))
        for info in engine_group_infos
        if info.layer_indices
    ):
        block_size, num_layers, hidden, dtype_str, kv_format, kv_size = (
            compute_kv_layout(
                {names[i]: kv_caches[names[i]] for i in layers},
                layout_hints=layout_hints,
            )
        )
        dtype = getattr(torch, dtype_str)
        geometry.add((block_size, kv_size))
        if num_layers != len(layers):
            reasons.append(
                f"group of layers {layers} has {len(layers)} layers but its "
                f"KV tensors hold {num_layers}"
            )
        groups.append(
            HybridGroupLayout(gid, layers, kv_format, dtype, hidden * dtype.itemsize)
        )
    if len(geometry) != 1:
        reasons.append(f"groups differ in (block size, K/V planes): {sorted(geometry)}")
    return groups, reasons


def _check_block_ids(block_ids: list[list[int]], layout: HybridLayout) -> None:
    """Reject block ids that do not match the registered engine groups."""
    if len(block_ids) != layout.num_engine_groups:
        raise ValueError(
            f"got block ids for {len(block_ids)} engine groups, registered "
            f"{layout.num_engine_groups}"
        )
    lengths = sorted({len(ids) for ids in block_ids})
    if len(lengths) != 1:
        raise ValueError(f"KV groups have different block counts: {lengths}")


def _chunk_blocks(
    block_ids: list[list[int]], gid: int, chunk_idx: int, blocks_per_chunk: int
) -> list[int]:
    """Return engine group ``gid``'s block ids for chunk ``chunk_idx``."""
    return block_ids[gid][
        chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk
    ]


def _layer_dim(chunk: torch.Tensor) -> int:
    """Return the layer dimension: split-K/V chunks are ``[2, L, T, H]``,
    single-plane chunks ``[L, T, H]``."""
    return 1 if chunk.dim() == 4 else 0


def _group_source(
    chunk: torch.Tensor,
    group: HybridGroupLayout,
    idx: torch.Tensor,
    byte_objects: bool,
) -> torch.Tensor:
    """Extract one group's layers of a full-layer chunk in the group's dtype."""
    if not byte_objects:
        return chunk.index_select(_layer_dim(chunk), idx)
    # index_select returns a contiguous tensor, which the dtype view requires.
    return (
        chunk.narrow(-1, 0, group.token_bytes)
        .index_select(_layer_dim(chunk), idx)
        .view(group.dtype)
    )


def _consecutive_runs(chunk_indices: list[int]) -> list[list[int]]:
    """Split ascending chunk indices into runs of consecutive indices."""
    runs: list[list[int]] = []
    for chunk_idx in chunk_indices:
        if runs and runs[-1][-1] == chunk_idx - 1:
            runs[-1].append(chunk_idx)
        else:
            runs.append([chunk_idx])
    return runs


@dataclass(frozen=True)
class HybridGroupLayout:
    """Transfer layout of one LMCache KV group's layers."""

    engine_group_id: int
    """Engine group the layers live in: the index of their request block ids."""

    layer_indices: list[int]
    """Registered KV tensor indices of this group, ascending."""

    engine_kv_format: "lmcache_native.EngineKVFormat"
    """KV format detected from this group's tensors."""

    dtype: torch.dtype
    """Element dtype of this group's KV tensors."""

    token_bytes: int
    """Bytes of one token of one layer in one K/V plane."""


@dataclass(frozen=True)
class HybridLayout:
    """Per-group transfer layout of a hybrid model."""

    num_engine_groups: int
    """Number of block-id lists per request."""

    groups: list[HybridGroupLayout]
    """One entry per LMCache KV group, sorted by ``engine_group_id``."""

    object_dtype: torch.dtype
    """Element dtype of the stored chunk objects."""

    object_hidden_dim: int
    """Per-token width of the stored chunk objects, in ``object_dtype``."""

    byte_objects: bool
    """Whether objects hold raw bytes because the groups' dtypes or widths
    differ; each layer's token row is then zero-padded to the widest group."""


def build_hybrid_layout(
    engine_group_infos: Sequence[EngineGroupInfo],
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None = None,
) -> HybridLayout | None:
    """Describe each KV group's transfer layout, or disable hybrid transfer.

    Request block ids are indexed by engine group id. One engine group may be
    split into several LMCache groups; each LMCache group is moved with its
    own detected layout and its engine group's block ids. Hybrid transfer is
    enabled only for layouts it moves exactly:

    * at least two engine groups, with dense ids ``0..G-1``;
    * every registered layer belongs to exactly one group (cross-layer
      KV-sharing layers that belong to no group are not supported);
    * one ``tokens_per_block`` for all groups, and the same detected block
      size and K/V plane count, so a chunk spans the same number of blocks
      and tokens in every group;
    * no sliding-window attention group. A one-block window
      (``sw_size_tokens == tokens_per_block``) is how align-mode Mamba /
      linear-attention layers are reported and is supported.

    The groups may differ in dtype and per-token width; the objects then hold
    raw bytes (see :attr:`HybridLayout.byte_objects`).

    Args:
        engine_group_infos: LMCache KV group metadata from registration.
        kv_caches: Registered KV tensors keyed by layer name.
        layout_hints: Engine layout hints used for registration.

    Returns:
        The hybrid layout, or ``None`` when hybrid transfer is not enabled, in
        which case multi-group transfers keep being rejected.
    """
    if num_engine_groups(engine_group_infos) <= 1:
        # Not hybrid: one block-id list per request, the regular path applies.
        return None
    reasons = _unsupported_group_reasons(engine_group_infos, len(kv_caches))
    groups: list[HybridGroupLayout] = []
    if not reasons:
        try:
            groups, reasons = _group_layouts(
                engine_group_infos, kv_caches, layout_hints
            )
        except ValueError as exc:
            reasons = [f"undetected group KV format ({exc})"]
    if reasons:
        logger.warning(
            "Hybrid engine-driven transfer disabled (%s); multi-group stores "
            "and retrieves stay unsupported.",
            "; ".join(reasons),
        )
        return None
    widths = {group.token_bytes for group in groups}
    byte_objects = len({group.dtype for group in groups}) > 1 or len(widths) > 1
    if byte_objects:
        object_dtype, object_hidden_dim = torch.uint8, max(widths)
    else:
        object_dtype = groups[0].dtype
        object_hidden_dim = groups[0].token_bytes // object_dtype.itemsize
    layout = HybridLayout(
        num_engine_groups(engine_group_infos),
        groups,
        object_dtype,
        object_hidden_dim,
        byte_objects,
    )
    logger.info(
        "Hybrid engine-driven transfer enabled: %d engine groups, %d KV groups, "
        "layers per group=%s, group dtypes=%s, bytes per token=%s, "
        "byte objects=%s",
        layout.num_engine_groups,
        len(groups),
        [len(group.layer_indices) for group in groups],
        [str(group.dtype).replace("torch.", "") for group in groups],
        [group.token_bytes for group in groups],
        layout.byte_objects,
    )
    return layout


def null_chunk_indices(
    block_ids: list[list[int]],
    layout: HybridLayout,
    blocks_per_chunk: int,
    chunk_indices: Sequence[int],
) -> list[int]:
    """Return the chunks that hold no valid KV in at least one engine group.

    Args:
        block_ids: Per-engine-group block ids of the request.
        layout: Output of :func:`build_hybrid_layout`.
        blocks_per_chunk: Engine blocks per LMCache chunk.
        chunk_indices: Chunks to check.

    Returns:
        The subset of ``chunk_indices`` that must not be stored.

    Raises:
        ValueError: If ``block_ids`` does not match the registered groups.
    """
    _check_block_ids(block_ids, layout)
    return [
        chunk_idx
        for chunk_idx in chunk_indices
        if any(
            all(
                block == NULL_BLOCK_ID
                for block in _chunk_blocks(block_ids, gid, chunk_idx, blocks_per_chunk)
            )
            for gid in range(layout.num_engine_groups)
        )
    ]


def gather_hybrid_paged_kv_to_cpu(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    layout: HybridLayout,
    blocks_per_chunk: int,
    layout_hints: LayoutHints | None = None,
    out: list[torch.Tensor] | None = None,
    chunk_indices: list[int] | None = None,
) -> list[torch.Tensor]:
    """Gather a hybrid model's paged KV blocks into full-layer CPU chunks.

    Same contract as :func:`gather_paged_kv_to_cpu`, except that each KV
    group's layers are read with its own format and its engine group's block
    ids, and the chunks use ``layout.object_dtype`` /
    ``layout.object_hidden_dim``. The current stream is synchronized before
    the per-group results are assembled on the host.

    Args:
        kv_caches: Registered KV tensors keyed by layer name.
        block_ids: Per-engine-group block ids of the request.
        layout: Output of :func:`build_hybrid_layout`.
        blocks_per_chunk: Engine blocks per LMCache chunk.
        layout_hints: Optional engine layout hints.
        out: Optional full-layer chunk tensors to fill: at least one per
            gathered chunk; extra buffers are ignored.
        chunk_indices: Optional chunk positions to gather; all chunks if
            ``None``.

    Returns:
        One full-layer CPU tensor per gathered chunk (``out`` when given).

    Raises:
        ValueError: If ``block_ids`` does not match the registered groups, or
            ``out`` has fewer buffers than the number of gathered chunks.
    """
    _check_block_ids(block_ids, layout)
    names = list(kv_caches)
    if chunk_indices is None:
        chunk_indices = list(range(len(block_ids[0]) // blocks_per_chunk))
    if out is not None and len(out) < len(chunk_indices):
        raise ValueError(
            f"out has {len(out)} buffers for {len(chunk_indices)} gathered chunks"
        )
    per_group: list[tuple[HybridGroupLayout, list[torch.Tensor]]] = []
    for group in layout.groups:
        chunks = gather_paged_kv_to_cpu(
            {names[i]: kv_caches[names[i]] for i in group.layer_indices},
            block_ids[group.engine_group_id],
            blocks_per_chunk,
            layout_hints=layout_hints,
            engine_kv_format=group.engine_kv_format,
            chunk_indices=chunk_indices,
        )
        per_group.append((group, chunks))
    # The device-to-host copies above may still be in flight.
    torch_dev.current_stream().synchronize()
    if out is None:
        first = per_group[0][1][0]
        shape = list(first.shape)
        shape[_layer_dim(first)] = len(names)
        shape[-1] = layout.object_hidden_dim
        out = [torch.empty(shape, dtype=layout.object_dtype) for _ in chunk_indices]
    for group, chunks in per_group:
        idx = torch.tensor(group.layer_indices, dtype=torch.long)
        # out may hold extra buffers beyond the gathered chunks; they are ignored.
        for dst, src in zip(out, chunks, strict=False):
            dim = _layer_dim(dst)
            if layout.byte_objects:
                src = src.view(torch.uint8)
                pad = dst.shape[-1] - src.shape[-1]
                if pad:
                    dst.narrow(-1, src.shape[-1], pad).index_fill_(dim, idx, 0)
                dst = dst.narrow(-1, 0, src.shape[-1])
            dst.index_copy_(dim, idx, src)
    return out[: len(chunk_indices)]


def scatter_hybrid_cpu_to_paged_kv(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    layout: HybridLayout,
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: LayoutHints | None = None,
) -> None:
    """Scatter full-layer CPU chunks back into a hybrid model's paged KV.

    Same contract as :func:`scatter_cpu_to_paged_kv`, except that each KV
    group's layers are written with its own format to its engine group's
    blocks, from chunks laid out as described by ``layout``. Chunks that are
    null in a group (align-mode Mamba keeps only the latest state block) or
    lie entirely inside the skipped prefix are not written for that group.
    The device is synchronized before returning, so the temporary per-group
    sources outlive their host-to-device copies.

    Args:
        kv_caches: Registered KV tensors keyed by layer name to write into.
        block_ids: Per-engine-group block ids of the request.
        layout: Output of :func:`build_hybrid_layout`.
        chunks: Full-layer CPU chunks, one per chunk of the request.
        blocks_per_chunk: Engine blocks per LMCache chunk.
        skip_first_n_tokens: Leading tokens of the range not to overwrite.
        layout_hints: Optional engine layout hints.

    Raises:
        ValueError: If ``block_ids`` does not match the registered groups.
    """
    if not chunks:
        return
    _check_block_ids(block_ids, layout)
    names = list(kv_caches)
    chunk_tokens = chunks[0].shape[-2]
    sources: list[torch.Tensor] = []
    for group in layout.groups:
        gid = group.engine_group_id
        idx = torch.tensor(group.layer_indices, dtype=torch.long)
        kept = [
            chunk_idx
            for chunk_idx in range(len(chunks))
            if (chunk_idx + 1) * chunk_tokens > skip_first_n_tokens
            and any(
                block != NULL_BLOCK_ID
                for block in _chunk_blocks(block_ids, gid, chunk_idx, blocks_per_chunk)
            )
        ]
        for run in _consecutive_runs(kept):
            run_sources = [
                _group_source(chunks[chunk_idx], group, idx, layout.byte_objects)
                for chunk_idx in run
            ]
            sources.extend(run_sources)
            scatter_cpu_to_paged_kv(
                {names[i]: kv_caches[names[i]] for i in group.layer_indices},
                [
                    block
                    for chunk_idx in run
                    for block in _chunk_blocks(
                        block_ids, gid, chunk_idx, blocks_per_chunk
                    )
                ],
                run_sources,
                blocks_per_chunk,
                skip_first_n_tokens=max(0, skip_first_n_tokens - run[0] * chunk_tokens),
                layout_hints=layout_hints,
                engine_kv_format=group.engine_kv_format,
            )
    torch_dev.synchronize()
