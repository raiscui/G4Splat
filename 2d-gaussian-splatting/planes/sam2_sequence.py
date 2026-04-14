from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
from typing import Iterator, Sequence


DEFAULT_VIEW_ORDER = (0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9)


@dataclass(frozen=True)
class InterleavedSequenceLayout:
    num_views: int = 12
    view_order: tuple[int, ...] = DEFAULT_VIEW_ORDER

    def __post_init__(self):
        normalized = tuple(int(v) for v in self.view_order)
        if len(normalized) != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} entries in view_order, got {len(normalized)}: {normalized}"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"view_order must not contain duplicates: {normalized}")
        object.__setattr__(self, "view_order", normalized)

    def split_index(self, index: int) -> tuple[int, int]:
        return index // self.num_views, index % self.num_views

    def frame_index(self, time_idx: int, view_slot: int) -> int:
        return time_idx * self.num_views + view_slot


@dataclass(frozen=True)
class SequenceFrame:
    global_frame_idx: int
    sequence_frame_idx: int
    time_idx: int
    view_slot: int
    source_view_id: int
    rgb_path: Path
    normal_path: Path | None

    @property
    def frame_idx(self) -> int:
        return self.global_frame_idx

    @property
    def view_id(self) -> int:
        return self.source_view_id


@dataclass(frozen=True)
class SequenceGroup:
    layout: str
    num_views: int
    view_order: tuple[int, ...]
    sequences: tuple[tuple[SequenceFrame, ...], ...]

    @property
    def total_frames(self) -> int:
        return sum(len(sequence) for sequence in self.sequences)


def _parse_frame_index(path: Path, prefix: str, suffix: str) -> int:
    name = path.name
    if not (name.startswith(prefix) and name.endswith(suffix)):
        raise ValueError(f"Unexpected frame name: {path}")
    return int(name[len(prefix) : -len(suffix)])


def _resolve_layout(path: Path, layout: str, num_views: int, rgb_count: int) -> str:
    if layout not in {"auto", "interleaved", "single"}:
        raise ValueError(f"Unsupported layout: {layout}")
    if layout != "auto":
        return layout
    if path.name == "select-gs-planes":
        return "single"
    if path.name == "plane-refine-depths":
        return "interleaved"
    if rgb_count <= num_views:
        return "single"
    return "interleaved"


def build_sequences(
    plane_root_path: str | Path,
    *,
    num_views: int = 12,
    view_order: Sequence[int] | None = None,
    layout: str = "interleaved",
) -> SequenceGroup:
    plane_root = Path(plane_root_path)
    rgb_paths = sorted(plane_root.glob("rgb_frame*.png"))
    if not rgb_paths:
        raise FileNotFoundError(f"No rgb_frame*.png found in {plane_root}")

    normal_lookup = {
        _parse_frame_index(path, "mono_normal_frame", ".npy"): path
        for path in plane_root.glob("mono_normal_frame*.npy")
    }

    layout_name = _resolve_layout(plane_root, layout, num_views, len(rgb_paths))
    normalized_view_order = (
        DEFAULT_VIEW_ORDER
        if view_order is None and num_views == len(DEFAULT_VIEW_ORDER)
        else tuple(range(num_views))
        if view_order is None
        else tuple(view_order)
    )

    if layout_name == "single":
        sequence = []
        for sequence_frame_idx, rgb_path in enumerate(rgb_paths):
            global_frame_idx = _parse_frame_index(rgb_path, "rgb_frame", ".png")
            sequence.append(
                SequenceFrame(
                    global_frame_idx=global_frame_idx,
                    sequence_frame_idx=sequence_frame_idx,
                    time_idx=sequence_frame_idx,
                    view_slot=0,
                    source_view_id=0,
                    rgb_path=rgb_path,
                    normal_path=normal_lookup.get(global_frame_idx),
                )
            )
        return SequenceGroup(layout="single", num_views=1, view_order=(0,), sequences=(tuple(sequence),))

    layout_config = InterleavedSequenceLayout(num_views=num_views, view_order=normalized_view_order)
    sequences = [[] for _ in range(layout_config.num_views)]
    for position, rgb_path in enumerate(rgb_paths):
        global_frame_idx = _parse_frame_index(rgb_path, "rgb_frame", ".png")
        time_idx, view_slot = layout_config.split_index(position)
        sequences[view_slot].append(
            SequenceFrame(
                global_frame_idx=global_frame_idx,
                sequence_frame_idx=len(sequences[view_slot]),
                time_idx=time_idx,
                view_slot=view_slot,
                source_view_id=layout_config.view_order[view_slot],
                rgb_path=rgb_path,
                normal_path=normal_lookup.get(global_frame_idx),
            )
        )

    return SequenceGroup(
        layout="interleaved",
        num_views=layout_config.num_views,
        view_order=layout_config.view_order,
        sequences=tuple(tuple(sequence) for sequence in sequences if sequence),
    )


@contextmanager
def materialize_sequence_jpgs(sequence_frames: Sequence[SequenceFrame]) -> Iterator[str]:
    if not sequence_frames:
        raise ValueError("Cannot materialize an empty sequence")

    with tempfile.TemporaryDirectory(prefix="sam2-sequence-") as temp_dir:
        for frame in sequence_frames:
            target = Path(temp_dir) / f"{frame.sequence_frame_idx:05d}.jpg"
            try:
                os.symlink(frame.rgb_path.resolve(), target)
            except OSError:
                target.write_bytes(frame.rgb_path.read_bytes())
        yield temp_dir


materialize_sequence_jpegs = materialize_sequence_jpgs
