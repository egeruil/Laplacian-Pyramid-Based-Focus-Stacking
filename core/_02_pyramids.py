"""
Build Gaussian and Laplacian pyramids for images. Refer to Burt & Adelson (1983) for more details.
Laplacian pyramids are constructed by subtracting the expanded version of the next level of the Gaussian pyramid from the current level.
"""

import cv2
import numpy as np
import os

def build_gaussian_pyramid(image, max_levels):
    """
    Build a Gaussian pyramid for a given image. 
    [G0, G1, ..., G{max_levels-1}]

    Args:
        image (np.ndarray): The input image.
        max_levels (int): The maximum number of levels in the pyramid.
    Returns:
        list: A list of images representing the Gaussian pyramid.
    """
    gaussian_pyramid = [image]
    for _ in range(max_levels):
        gaussian_next = cv2.pyrDown(gaussian_pyramid[-1])
        gaussian_pyramid.append(gaussian_next)

    return gaussian_pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    """
    Build a Laplacian pyramid from a given Gaussian pyramid.
    [L0, L1, ..., L{max_levels-2}, G{max_levels-1}]

    Args:
        gaussian_pyramid (list): A list of images representing the Gaussian pyramid.
    Returns:
        list: A list of images representing the Laplacian pyramid.
    """
    laplacian_pyramid = []
    num_levels = len(gaussian_pyramid)

    for k in range(num_levels - 1):
        gauss_k = gaussian_pyramid[k]
        gauss_k_plus_1 = gaussian_pyramid[k + 1]
        
        # Upsample the next level to match the current level's size
        gauss_k_plus_1_up = cv2.pyrUp(gauss_k_plus_1, dstsize=(gauss_k.shape[1], gauss_k.shape[0]))
        # print(gauss_k.shape, gauss_k_plus_1_up.shape)
        
        laplacian = gauss_k - gauss_k_plus_1_up
        laplacian_pyramid.append(laplacian)

    return laplacian_pyramid, gaussian_pyramid[-1]

def build_pyramids_stack(images, levels, gaussian_pyramid_dir=None, laplacian_pyramid_dir=None):
    """
    Build Gaussian and Laplacian pyramids for a stack of images.

    Args:
        images (np.ndarray): A 3D numpy array containing the stacked images.
        levels (int): The number of levels in the pyramids.
    Returns:
        tuple: A tuple containing two lists:
            - gaussian_pyramids: A list of Gaussian pyramids for each image.
            - laplacian_pyramids: A list of Laplacian pyramids for each image.
    """
    top_gaussians = []
    gaussian_pyramids = []
    laplacian_pyramids = []

    for image in images:
        gaussian_pyramid = build_gaussian_pyramid(image, levels)
        laplacian_pyramid, top_gaussian = build_laplacian_pyramid(gaussian_pyramid)

        top_gaussians.append(top_gaussian)
        gaussian_pyramids.append(gaussian_pyramid)
        laplacian_pyramids.append(laplacian_pyramid)

    # save pyramids if directories are provided
    if gaussian_pyramid_dir is not None:
        os.makedirs(gaussian_pyramid_dir, exist_ok=True)
        for i, gpyr in enumerate(gaussian_pyramids):
            image_dir = os.path.join(gaussian_pyramid_dir, f"image_{i:03d}")
            os.makedirs(image_dir, exist_ok=True)
            for k, level in enumerate(gpyr):
                cv2.imwrite(os.path.join(image_dir, f"level_{k:02d}.png"), level)

    if laplacian_pyramid_dir is not None:
        os.makedirs(laplacian_pyramid_dir, exist_ok=True)
        for i, lpyr in enumerate(laplacian_pyramids):
            image_dir = os.path.join(laplacian_pyramid_dir, f"image_{i:03d}")
            os.makedirs(image_dir, exist_ok=True)
            for k, level in enumerate(lpyr):
                cv2.imwrite(os.path.join(image_dir, f"level_{k:02d}.png"), level + 128)  # shift for visualization
        
    return gaussian_pyramids, laplacian_pyramids, top_gaussians