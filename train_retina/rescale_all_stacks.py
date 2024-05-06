import os
from PIL import Image
import tifffile as tiff
import numpy as np
from skimage import exposure
from skimage import img_as_uint

def rescale_images(source_dir, target_dir, size=(512, 512)):
    """
    Rescales the images in the source directory and saves the rescaled images in the target directory.

    Args:
        source_dir (str): The path to the source directory containing the images.
        target_dir (str): The path to the target directory where the rescaled images will be saved.
        size (tuple, optional): The desired size of the rescaled images. Defaults to (512, 512).
    """
    subfolders = ['FG12', 'PBS', 'NG004', '11C7']
    for subfolder in subfolders:
        subfolder_path = os.path.join(source_dir, subfolder)
        
        tiff_files = [f for f in os.listdir(subfolder_path) if f.endswith('.tif') and not f.endswith('_contrast.tif')]
        for filename in tiff_files:
            print(f'Processing {filename}...')
            img_path = os.path.join(subfolder_path, filename)
            imgs = tiff.imread(img_path)  
            
            for slice_index, img in enumerate(imgs):
                img = exposure.equalize_hist(img)
                
                img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                img_normalized = img_normalized.astype('uint8')
                
                img_rescaled = Image.fromarray(img_normalized).resize(size, Image.Resampling.LANCZOS)
                img_rescaled = img_rescaled.convert('L') 
                
                # Create target directory path for the current slice
                target_subfolder_path = os.path.join(target_dir, subfolder, os.path.splitext(filename)[0])
                if not os.path.exists(target_subfolder_path):
                    os.makedirs(target_subfolder_path)
                
                # Save the rescaled image slice
                target_file_path = os.path.join(target_subfolder_path, f'{os.path.splitext(filename)[0]}_slice_{slice_index}.png')
                img_rescaled.save(target_file_path)
                print(f'Slice {slice_index} of {filename} rescaled and saved.')

os.chdir('N:\\00_Exchange\\Alois')
source_dir = 'data_full'
target_dir = 'data_retina_full/rescaled_retina'
rescale_images(source_dir, target_dir)
