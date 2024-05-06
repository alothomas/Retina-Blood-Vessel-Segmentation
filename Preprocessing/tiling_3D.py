import numpy as np
from PIL import Image
import os
from tifffile import imread
from skimage import exposure

def extract_overlapping_patches(image, patch_size=512, overlap=0.3):
    """
    Extracts overlapping patches from an image.

    Parameters:
    - image: numpy.ndarray
        The input image from which patches will be extracted.
    - patch_size: int, optional
        The size of each patch. Default is 512.
    - overlap: float, optional
        The overlap between patches as a fraction of the patch size. Default is 0.3.

    Returns:
    - patches: list
        A list of extracted patches from the image.
    """
    stride = int(patch_size * (1 - overlap))
    patches = []
    for y in range(0, image.shape[0] - patch_size + 1, stride):
        for x in range(0, image.shape[1] - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches

def apply_contrast_adjustments(patch):
    patch_rescaled = exposure.rescale_intensity(patch, in_range='image')
    patch_eq = exposure.equalize_adapthist(patch_rescaled, clip_limit=0.15)
    return patch_eq

def normalize_patch(patch):
    patch_min = patch.min()
    patch_max = patch.max()
    normalized_patch = np.zeros_like(patch) if patch_max == patch_min else (patch - patch_min) / (patch_max - patch_min)
    return normalized_patch

# Change to your base directory
os.chdir('N:\\00_Exchange\\Alois')

input_dirs = [r'data_full/11C7', r'data_full/FG12', r'data_full/NG004', r'data_full/PBS']
output_base = 'data_full/data_patch'

patch_size = 1024
overlap = 0.3

for input_dir in input_dirs:
    dir_name = os.path.basename(input_dir)  # Get the name of the current directory
    filenames = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    
    for filename in filenames:
        file_path = os.path.join(input_dir, filename)
        multi_layer_image = imread(file_path) 
        
        # Extract the base filename without extension to use in output path
        base_filename = os.path.splitext(filename)[0]
        
        for layer_index, layer in enumerate(multi_layer_image):
            patches = extract_overlapping_patches(layer, patch_size, overlap)
            
            for i, patch in enumerate(patches):
                contrast_adjusted_patch = apply_contrast_adjustments(patch)
                normalized_patch = normalize_patch(contrast_adjusted_patch)
                patch_image = Image.fromarray((normalized_patch * 255).astype(np.uint8))
                
                # Adjust output directory for each layer to include subfolder for image name
                output_dir_for_image = os.path.join(output_base, dir_name, base_filename)
                output_dir_for_layer = os.path.join(output_dir_for_image, f"layer_{layer_index}")
                
                if not os.path.exists(output_dir_for_layer):
                    os.makedirs(output_dir_for_layer)
                    
                patch_filename = f'patch_{i}.png'
                patch_image.save(os.path.join(output_dir_for_layer, patch_filename), 'PNG')


print('Done')