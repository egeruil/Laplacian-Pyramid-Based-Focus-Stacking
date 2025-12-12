"""
Baseline: use absolute of laplacian as sharpness metric
"""

import cv2
import numpy as np

def compute_sharpness_map(laplacian_pyramids, output_dir=None):
    """
    Compute the sharpness map from the Laplacian pyramid.

    Args:
        laplacian_pyramids (list): A list of Laplacian pyramids for all images.
            - len(laplacian_pyramids)   = num_images
            - len(laplacian_pyramids[0]) = num_levels
            Each laplacian_pyramids[i] is a list: [L0, L1, ..., L{L-1}],
            where Lk is a 2D array (H_k, W_k) for level k.

    Returns:
        sharpness_maps (list[list[np.ndarray]]):
            A list of lists containing the sharpness maps for each level
            of the Laplacian pyramid.
            - sharpness_maps[i][k] has the same shape as laplacian_pyramids[i][k],
              and represents the sharpness of image i at level k.
    """

    num_images = len(laplacian_pyramids)
    if num_images == 0:
        return []

    num_levels = len(laplacian_pyramids[0])

    sharpness_maps = []

    for i in range(num_images):
        lap_pyr = laplacian_pyramids[i]
        level_sharpness = []

        for k in range(num_levels):
            Lk = lap_pyr[k]

            # Compute sharpness metric
            # Using Gaussian smoothed squared Laplacian (local energy)
            # For color images, this produces a per-channel sharpness map
            Ek = cv2.GaussianBlur(Lk * Lk, (3, 3), 0)
            
            level_sharpness.append(Ek)

        sharpness_maps.append(level_sharpness)

    # print("Sharpness maps shape:", [[sharpness_maps[i][k].shape for k in range(num_levels)] for i in range(num_images)])

    # save sharpness maps for debugging if output_dir is provided
    if output_dir is not None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        for i in range(num_images):
            for k in range(num_levels):
                sharp_map = sharpness_maps[i][k]
                # Normalize for visualization
                sharp_map_norm = cv2.normalize(sharp_map, None, 0, 255, cv2.NORM_MINMAX)
                sharp_map_uint8 = sharp_map_norm.astype(np.uint8)
                output_path = os.path.join(output_dir, f"image_{i}_level_{k}_sharpness.png")
                cv2.imwrite(output_path, sharp_map_uint8)

    return sharpness_maps

