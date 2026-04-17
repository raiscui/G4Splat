from __future__ import annotations

import os

import numpy as np
import torch


def build_depth_agreement_mask(
    *,
    warp_depth: torch.Tensor,
    aligned_depth: torch.Tensor,
    candidate_mask: torch.Tensor,
    max_relative_error: float,
    max_absolute_error: float,
) -> torch.Tensor:
    candidate_mask = candidate_mask.bool()
    valid_depths = (
        torch.isfinite(warp_depth)
        & torch.isfinite(aligned_depth)
        & (warp_depth > 0)
        & (aligned_depth > 0)
    )
    valid_mask = candidate_mask & valid_depths
    if not torch.any(valid_mask):
        return valid_mask

    depth_diff = (warp_depth - aligned_depth).abs()
    relative_scale = torch.maximum(warp_depth.abs(), aligned_depth.abs()) * float(max_relative_error)
    allowed_error = torch.maximum(relative_scale, torch.full_like(depth_diff, float(max_absolute_error)))
    return valid_mask & (depth_diff <= allowed_error)


def erode_binary_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    mask = mask.bool()
    if radius <= 0:
        return mask
    kernel_size = 2 * int(radius) + 1
    mask_f = mask.float().unsqueeze(0).unsqueeze(0)
    eroded = 1.0 - torch.nn.functional.max_pool2d(
        1.0 - mask_f,
        kernel_size=kernel_size,
        stride=1,
        padding=int(radius),
    )
    return eroded[0, 0] > 0.5


def load_confident_mask_from_visibility(
    *,
    plane_root_path: str,
    view_id: int,
    fallback_shape: tuple[int, int],
) -> np.ndarray:
    visibility_path = os.path.join(plane_root_path, f"visibility_frame{view_id:06d}.npy")
    if os.path.exists(visibility_path):
        visibility = np.load(visibility_path)
        return (visibility > 0.5).astype(np.uint8)

    return np.ones(fallback_shape, dtype=np.uint8)
