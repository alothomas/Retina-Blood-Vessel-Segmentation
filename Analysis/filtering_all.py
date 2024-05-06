import os
import tifffile
import numpy as np
import cv2
from scipy.ndimage import uniform_filter

def apply_custom_filter_with_mask_convolution_optimized(image_array, window_size, threshold_percentage):
    """
    Apply a custom filter with mask convolution to the input image array.

    Parameters:
    image_array (ndarray): The input image array.
    window_size (tuple): The size of the window for convolution.
    threshold_percentage (float): The threshold percentage for filtering.

    Returns:
    ndarray: The mask indicating the removed windows.

    """
    threshold = window_size[0] * window_size[1] * threshold_percentage
    binary_mask = (image_array == 255).astype(np.float32)
    white_pixels_count = uniform_filter(binary_mask, size=window_size, mode='constant', cval=0)
    removed_windows_mask = white_pixels_count * window_size[0] * window_size[1] >= threshold
    
    return removed_windows_mask

def process_image(image_path, binary_image_path, output_dir, subject_name, window_size, threshold_percentage):
    """
    Process the input image and perform blood vessel segmentation.

    Args:
        image_path (str): The path to the original image.
        binary_image_path (str): The path to the binary image.
        output_dir (str): The directory to save the processed images.
        subject_name (str): The name of the subject.
        window_size (int): The size of the window for filtering.
        threshold_percentage (float): The threshold percentage for filtering.

    Returns:
        None
    """
    print(f'Processing image {subject_name}')

    layers_vessel = []
    with tifffile.TiffFile(binary_image_path) as tif:
        for i, page in enumerate(tif.pages):
            layer = page.asarray()
            layers_vessel.append(layer)

    binary_image = np.stack(layers_vessel)
    

    with tifffile.TiffFile(image_path) as tif:
        original_pages = [page.asarray() for page in tif.pages]
    original_image = np.stack(original_pages)


    # Normalize to 8-bit if needed
    original_image_8bit = np.array([((layer - layer.min()) / (layer.max() - layer.min()) * 255).astype(np.uint8) if layer.dtype != np.uint8 else layer for layer in original_image])
    binary_image = np.array([((layer - layer.min()) / (layer.max() - layer.min()) * 255).astype(np.uint8) if layer.dtype != np.uint8 else layer for layer in binary_image])

    # Apply threshold to each layer of the multi-layer image
    binary_image_thresholded = np.array([cv2.threshold(layer, 127, 255, cv2.THRESH_BINARY)[1] for layer in binary_image])

    removed_windows_masks = [apply_custom_filter_with_mask_convolution_optimized(binary_image_thresholded[i], window_size, threshold_percentage) for i in range(binary_image_thresholded.shape[0])]
    removed_windows_mask = np.stack(removed_windows_masks)

    # Apply filter based on the removed windows mask
    original_image_filtered = np.zeros_like(original_image_8bit)
    for i in range(original_image_8bit.shape[0]):
        if removed_windows_mask[i].shape == original_image_8bit[i].shape:
            original_image_filtered[i][removed_windows_mask[i]] = 255
        else:
            print(f'Error: removed_windows_mask shape {removed_windows_mask.shape[i:]} does not match original_image_8bit shape {original_image_8bit[i].shape}')
    
    
    original_filtered_output_path = os.path.join(output_dir, f'{subject_name}_retina_mask_filtered.tif')
    tifffile.imwrite(original_filtered_output_path, original_image_filtered)

    binary_image_filtered = np.copy(binary_image_thresholded)
    for i in range(binary_image_filtered.shape[0]):
        binary_image_filtered[i][~removed_windows_mask[i]] = 0

    
    binary_image_output_path = os.path.join(output_dir, f'{subject_name}_binary_filtered.tif')
    tifffile.imwrite(binary_image_output_path, binary_image_filtered)

def process_directory(base_directory, data_processed_path, retina_masks_path):
    for condition in os.listdir(data_processed_path):
        condition_path = os.path.join(data_processed_path, condition)
        for subject_folder in os.listdir(condition_path):
            subject_path = os.path.join(condition_path, subject_folder)
            image_name_prefix = subject_folder.split('_')[0]
            binary_image_path = os.path.join(subject_path, f'{subject_folder}_reconstructed_multi.tif')
            image_path = os.path.join(retina_masks_path, condition, f'{subject_folder}_mask.tif')
            process_image(image_path, binary_image_path, subject_path, image_name_prefix, (200, 200), 0.05)

base_directory = 'N:\\00_Exchange\\Alois\\data_full'
data_processed_path = os.path.join(base_directory, 'data_processed')
retina_masks_path = os.path.join(base_directory, 'retina_masks')

process_directory(base_directory, data_processed_path, retina_masks_path)

print('Done!')
