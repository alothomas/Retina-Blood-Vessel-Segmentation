from skimage import exposure
import numpy as np
import czifile, tifffile
import os
from skimage import io

def process_images(image_dir, output_dir):
    """
    Process CZI images in the specified directory and extract layers.
    
    Args:
        image_dir (str): The directory containing CZI image files.
        output_dir (str): The directory to save the extracted layers as multi-page TIFF files.
    
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # List all CZI files in the directory
    image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.czi')]

    for image_file in image_paths:
        try:
            print(f"Extracting layers from {image_file}")
            with czifile.CziFile(image_file) as czi:
                image_array = czi.asarray()
                
            # Assuming the CZI image stack is in the 5th dimension
            layers = [image_array[0, 0, 0, 0, stack_index, :, :, 0] for stack_index in range(image_array.shape[4])]
            # Convert list of layers to a numpy array for saving as multi-page TIFF
            layers_array = np.stack(layers, axis=0)
            
            # Create output filename based on the CZI filename
            output_filename = os.path.basename(image_file).replace(".czi", ".tif")
            tifffile.imsave(os.path.join(output_dir, output_filename), layers_array, imagej=True)
            
            print(f"Saved all layers to {os.path.join(output_dir, output_filename)}")
                
        except Exception as e:
            print(f"Error processing file {image_file}: {e}")

os.chdir('N:\\00_Exchange\\Alois')

image_dirs = [r'N:\\00_Exchange\\Alois\\Akita\\Raw data groups\\11C7', 
              r'N:\\00_Exchange\\Alois\\Akita\\Raw data groups\\FG12',
              r'N:\\00_Exchange\\Alois\\Akita\\Raw data groups\\NG004',
              r'N:\\00_Exchange\\Alois\\Akita\\Raw data groups\\PBS']

output_dirs = [r'data_full/11C7', r'data_full/FG12', r'data_full/NG004', r'data_full/PBS']

for image_dir, output_dir in zip(image_dirs, output_dirs):
    process_images(image_dir, output_dir)

print("Done!")
