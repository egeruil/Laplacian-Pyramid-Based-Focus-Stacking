import os
import cv2
import numpy as np

from _01_preprocess import preprocess_image_stack
from _02_pyramids import build_pyramids_stack
from _03_sharpness import compute_sharpness_map
from _04_mask import build_masks
from _05_fusion import fuse_pyramids_and_reconstruct

def main(name):
    data_dir = os.path.join("../data", name)
    base_name = name

    print("Preprocessing image stack...")
    images = preprocess_image_stack(data_dir)
    levels = 4      # number of pyramid levels, can be adjusted

    # Build pyramids
    print("Building pyramids...")
    GAUSSIAN_PYR_DIR = "../output/gaussian_pyramids"
    LAPLACIAN_PYR_DIR = "../output/laplacian_pyramids"
    GAUSSIAN_PYR_DIR = os.path.join(GAUSSIAN_PYR_DIR, base_name)
    LAPLACIAN_PYR_DIR = os.path.join(LAPLACIAN_PYR_DIR, base_name)
    gaussian_pyrs, laplacian_pyrs, top_gaussians = build_pyramids_stack(
        images, levels, gaussian_pyramid_dir=GAUSSIAN_PYR_DIR, laplacian_pyramid_dir=LAPLACIAN_PYR_DIR)

    # Compute sharpness maps
    print("Computing sharpness maps...")
    SHARP_MAP_DIR = "../output/sharpness_maps"
    SHARP_MAP_DIR = os.path.join(SHARP_MAP_DIR, base_name)
    sharpness_maps = compute_sharpness_map(laplacian_pyrs, output_dir=SHARP_MAP_DIR)

    # Build masks
    print("Building decision masks...")
    smoothed_masks = build_masks(sharpness_maps, sigma=1.2, ksize=7)

    # Fuse pyramids and reconstruct
    print("Fusing pyramids and reconstructing fused image...")
    LAPLACIAN_LEV_and_TOP_GAUSSIAN_DIR = "../output/fused_pyramids"
    LAPLACIAN_LEV_and_TOP_GAUSSIAN_DIR = os.path.join(LAPLACIAN_LEV_and_TOP_GAUSSIAN_DIR, base_name)
    fused_image = fuse_pyramids_and_reconstruct(
        laplacian_pyrs, top_gaussians, smoothed_masks, top_fusion_method="max", output_dir=LAPLACIAN_LEV_and_TOP_GAUSSIAN_DIR)

    OUT_DIR = "../output/fused_images"
    os.makedirs(OUT_DIR, exist_ok=True)
    output_path = os.path.join(OUT_DIR, f"{base_name}_fused.png")
    cv2.imwrite(output_path, fused_image.astype(np.uint8))
    print(f"Saving fused image to {output_path}")
    
if __name__ == "__main__":
    name = input("Enter image folder name: ")
    main(name)