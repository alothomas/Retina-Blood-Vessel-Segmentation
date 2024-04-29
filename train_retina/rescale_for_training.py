import os
import random
from PIL import Image
import tifffile as tiff
import numpy as np

random.seed(42)

def rescale_images(source_dir, target_dir, size=(512, 512)):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    subfolders = ['FG12', 'PBS']
    image_counter = 1  
    for subfolder in subfolders:
        subfolder_path = os.path.join(source_dir, subfolder)
        tiff_files = [f for f in os.listdir(subfolder_path) if f.endswith('.tif')]
        random_files = random.sample(tiff_files, min(10, len(tiff_files)))

        for filename in random_files:
            print(f'Rescaling image {filename}...')
            img_path = os.path.join(subfolder_path, filename)
            img = tiff.imread(img_path)

            # Normalize the image data to 0-255 range
            img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
            img_normalized = img_normalized.astype('uint8')

            img = Image.fromarray(img_normalized)
            img_rescaled = img.resize(size, Image.Resampling.LANCZOS)
            img_rescaled = img_rescaled.convert('L')
            target_file_path = os.path.join(target_dir, f'image_{image_counter}.png')
            img_rescaled.save(target_file_path)

            image_counter += 1  

source_dir = 'data'
target_dir = 'data_retina/annotations'
rescale_images(source_dir, target_dir)
