"""
Preprocess module: reads raw data of different depth images from a specified folder
"""

import cv2
import numpy as np
import os
import glob

def load_image_stack(folder_path, file_extension='png'):
    """
    Load a stack of images from the specified folder.

    Args:
        folder_path (str): Path to the folder containing images.
        file_extension (str): Extension of the image files to load.
    
    Returns:
        np.ndarray: A 3D numpy array containing the stacked images.
    """
    image_files = sorted(glob.glob(os.path.join(folder_path, f'*.{file_extension}')))
    image_stack = []
    for image_file in image_files:
        image = cv2.imread(image_file, cv2.IMREAD_COLOR).astype(np.float32)
        if image is not None:
            image_stack.append(image)
            
    if image_stack:
        return np.stack(image_stack, axis=0)    # (N, H, W, C=3)
    else:
        return np.array([])
    
def ensure_same_size(image_stack):
    """
    Ensure all images in the stack have the same size by resizing them to the size of the first image.

    Args:
        image_stack (np.ndarray): A 3D numpy array containing the stacked images.
    Returns:
        np.ndarray: A 3D numpy array with all images resized to the same dimensions.
    """
    # print("Image stack size:", image_stack.size)

    if image_stack.size == 0:
        return image_stack
    
    target_shape = image_stack[0].shape
    resized_stack = []
    for image in image_stack:
        if image.shape != target_shape:
            resized_image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            resized_stack.append(resized_image)
        else:
            resized_stack.append(image)

    return np.stack(resized_stack, axis=0)

def align_images(image_stack):
    """
    Align images in the stack using ECC (Enhanced Correlation Coefficient) maximization.
    This is more robust than simple center-of-mass alignment.

    Args:
        image_stack (np.ndarray): A stack of images, shape (N, H, W[, C])
    Returns:
        np.ndarray: A stack with aligned images.
    """
    if image_stack.size == 0:
        return image_stack

    aligned_stack = []

    # Use the first image as the reference
    reference_image = image_stack[0]
    aligned_stack.append(reference_image)

    # Convert reference to grayscale for ECC
    if reference_image.ndim == 3 and reference_image.shape[2] > 1:
        ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = reference_image

    H, W = ref_gray.shape[:2]

    # Define the motion model
    # MOTION_AFFINE handles translation, rotation, scale, and shear
    warp_mode = cv2.MOTION_AFFINE

    # Set termination criteria
    number_of_iterations = 500
    termination_eps = 1e-5
    # Criteria: either 500 iterations or epsilon of 1e-5
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    for i in range(1, len(image_stack)):
        image = image_stack[i]
        
        # Convert to grayscale for ECC
        if image.ndim == 3 and image.shape[2] > 1:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image

        # Initialize warp matrix
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            # Run the ECC algorithm. The results are stored in warp_matrix.
            # findTransformECC finds the transform that maps the input image (img_gray) to the template (ref_gray)
            (_, warp_matrix) = cv2.findTransformECC(ref_gray, img_gray, warp_matrix, warp_mode, criteria)
            
            # Use warpAffine with the calculated matrix.
            aligned_image = cv2.warpAffine(
                image, 
                warp_matrix, 
                (W, H), 
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
        except cv2.error as e:
            print(f"Alignment failed for image {i}, keeping original. Error: {e}")
            aligned_image = image

        aligned_stack.append(aligned_image)

    return np.stack(aligned_stack, axis=0)


def preprocess_image_stack(folder_path, file_extension='png', use_cache=True):
    """
    Load, resize, and align images from a folder.
    Supports caching to speed up subsequent runs.
    """
    # Determine cache path
    base_name = os.path.basename(os.path.normpath(folder_path))
    # Cache directory inside the core folder
    cache_dir = os.path.join(os.path.dirname(__file__), "cache")
    cache_file = os.path.join(cache_dir, f"{base_name}_aligned.npy")

    if use_cache and os.path.exists(cache_file):
        print(f"Loading preprocessed images from cache: {cache_file}")
        try:
            return np.load(cache_file)
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    image_stack = load_image_stack(folder_path, file_extension)

    if image_stack.size == 0:
        raise ValueError(f"No images found in {folder_path} with extension .{file_extension}")

    image_stack = ensure_same_size(image_stack)
    image_stack = align_images(image_stack)

    if use_cache:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_file, image_stack)
            print(f"Saved preprocessed images to cache: {cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    return image_stack
 