"""
Fuse Laplacian pyramids and top Gaussian layers using the
smoothed decision masks, then reconstruct the final all-in-focus image.
"""

import cv2
import numpy as np
import os

def fuse_laplacian_pyramids(laplacian_pyramids, smoothed_masks, output_dir=None):
    num_images = len(laplacian_pyramids)
    if num_images == 0:
        return []

    num_levels = len(laplacian_pyramids[0])

    fused_laplacian = []
    for k in range(num_levels):
        # Allow (H, W) or (H, W, C)
        shape = laplacian_pyramids[0][k].shape
        if len(shape) == 2:
            H, W = shape
            C = None
        else:
            H, W, C = shape

        # Initialize fused Lk
        if C is None:
            Lk_fused = np.zeros((H, W), dtype=np.float32)
        else:
            Lk_fused = np.zeros((H, W, C), dtype=np.float32)

        for i in range(num_images):
            Lk = laplacian_pyramids[i][k].astype(np.float32)
            Wk = smoothed_masks[i][k].astype(np.float32)
            
            # If Lk is 3D (H, W, C) and Wk is 2D (H, W), expand Wk to (H, W, 1)
            if Lk.ndim == 3 and Wk.ndim == 2:
                Wk = Wk[:, :, np.newaxis]

            Lk_fused += Lk * Wk

        fused_laplacian.append(Lk_fused)
        # save fused laplacian level for debugging if output_dir is provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fused_level = Lk_fused
            # Normalize for visualization
            fused_level_norm = cv2.normalize(fused_level, None, 0, 255, cv2.NORM_MINMAX)
            fused_level_uint8 = fused_level_norm.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, f"fused_laplacian_level_{k}.png"), fused_level_uint8)

    return fused_laplacian


def fuse_top_gaussian(top_gaussians, method="mean", output_dir=None):
    """
    Fuse the top-level Gaussian images (lowest-frequency components).

    Args:
        top_gaussians (list[np.ndarray]):
            top_gaussians[i] = top-level Gaussian (lowest resolution) of image i.
        method (str): simple strategy like "mean" or "max", default is "mean".

    Returns:
        fused_top (np.ndarray):
            fused top-level Gaussian image.
    """
    if len(top_gaussians) == 0:
        return None

    stack = np.stack(top_gaussians, axis=0).astype(np.float32)  # (N, H, W)

    if method == "max":
        fused_top = np.max(stack, axis=0)
    else:
        # default: mean
        fused_top = np.mean(stack, axis=0)

    # save fused top gaussian for debugging if output_dir is provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fused_top_norm = cv2.normalize(fused_top, None, 0, 255, cv2.NORM_MINMAX)
        fused_top_uint8 = fused_top_norm.astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"fused_top_gaussian.png"), fused_top_uint8)

    return fused_top


def reconstruct_from_pyramid(fused_laplacian, fused_top):
    if fused_top is None:
        return None

    current = fused_top.astype(np.float32)
    num_levels = len(fused_laplacian)

    for k in reversed(range(num_levels)):
        Lk = fused_laplacian[k].astype(np.float32)

        # Allow 2D or 3D
        H, W = Lk.shape[:2]
        up = cv2.pyrUp(current, dstsize=(W, H))
        current = up + Lk

    fused_image = np.clip(current, 0.0, 255.0)
    return fused_image

def fuse_pyramids_and_reconstruct(laplacian_pyramids, top_gaussians, smoothed_masks, top_fusion_method="mean", output_dir=None):
    fused_laplacian = fuse_laplacian_pyramids(laplacian_pyramids, smoothed_masks, output_dir=output_dir)
    fused_top = fuse_top_gaussian(top_gaussians, method=top_fusion_method, output_dir=output_dir)
    fused_image = reconstruct_from_pyramid(fused_laplacian, fused_top)
    return fused_image