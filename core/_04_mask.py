"""
Build decision masks from sharpness maps and smooth them
for multi-focus (multi-image) fusion.
"""

import cv2
import numpy as np


def build_raw_masks(sharpness_maps):
    """
    Build raw (hard) decision masks from sharpness maps.
    """
    num_images = len(sharpness_maps)
    if num_images == 0:
        return []
    
    num_levels = len(sharpness_maps[0])

    # Initialize raw_masks: a list for each image, initially containing None for each level
    raw_masks = [[None] * num_levels for _ in range(num_images)]

    # Process level by level
    for k in range(num_levels):
        # Stack all images' sharpness maps at level k: (N, H_k, W_k)
        Ek_stack = np.stack(
            [sharpness_maps[i][k] for i in range(num_images)],
            axis=0
        )

        # Find the index of the image with the maximum sharpness for each pixel
        idx_max = np.argmax(Ek_stack, axis=0)  # shape: (H_k, W_k)

        # Create a one-hot mask for each image
        for i in range(num_images):
            mask = (idx_max == i).astype(np.float32)  # (H_k, W_k), 0/1
            raw_masks[i][k] = mask

    return raw_masks


def smooth_and_normalize_masks(raw_masks, sigma=1.0, ksize=5):
    """
    Smooth raw masks (edge-preserving at a basic level) and normalize
    so that sum_i W_i^k(x,y) == 1 (approximately).

    Args:
        raw_masks (list[list[np.ndarray]]):
            raw_masks[i][k] = the raw 0/1 mask of image i at level k, shape (H_k, W_k).
        sigma (float): std of Gaussian blur
        ksize (int): size of Gaussian kernel (must be odd).

    Returns:
        smoothed_masks (list[list[np.ndarray]]):
            smoothed_masks[i][k] = the smoothed and normalized mask of image i at level k,
            with values approximately in [0,1], and for each (x,y,k), sum_i smoothed_masks[i][k](x,y) ≈ 1.
    """
    num_images = len(raw_masks)
    if num_images == 0:
        return []

    num_levels = len(raw_masks[0])

    smoothed_masks = []
    for i in range(num_images):
        smoothed_masks.append([None] * num_levels)

    for k in range(num_levels):
        # First, apply Gaussian blur to each image's mask at this level
        blurred_list = []
        for i in range(num_images):
            m = raw_masks[i][k]
            # Ensure the mask is not empty and convert type to float32
            m = m.astype(np.float32)
            # Gaussian blur to avoid hard edges causing artifacts like jaggedness or halos during reconstruction
            mb = cv2.GaussianBlur(m, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
            blurred_list.append(mb)

        # Stack into (N, H, W)
        stack = np.stack(blurred_list, axis=0)

        # Normalize along the 0th dimension (image index)
        denom = np.sum(stack, axis=0, keepdims=True) + 1e-8  # Avoid division by zero
        norm_stack = stack / denom

        # Unpack back to list[list[np.ndarray]]
        for i in range(num_images):
            smoothed_masks[i][k] = norm_stack[i]

    return smoothed_masks

def build_masks(sharpness_maps, sigma=1.0, ksize=5):
    raw_masks = build_raw_masks(sharpness_maps)
    smoothed_masks = smooth_and_normalize_masks(raw_masks, sigma=sigma, ksize=ksize)
    return smoothed_masks