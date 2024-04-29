import os
import re
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tifffile import imread, imsave
import tifffile
from tifffile import TiffWriter
from skimage import exposure

from resnet50_unet import UNetWithResnet50Encoder
from utils_pred import PredCustomDataset, reconstruct_image_from_patches

# Setup device for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Model path
model_path = r'E:\Retina-Segmentation\resnetunet\output_dir\best_model_DRIVE_tuned152.pth'

# Load the model
model = UNetWithResnet50Encoder(n_classes=1).to(device)
model.load_state_dict(torch.load(model_path))

# Predict function
def predict_image(loader, model, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            output = model(images)
            output = torch.sigmoid(output)
            pred = output > 0.5
            pred = pred.squeeze(1)
            preds.extend(pred.detach().cpu().numpy())
    return preds



def process_folders(input_folders, output_base, model_path, patch_size=512, overlap=0.3, batch_size=8):
    for input_folder in input_folders:
        group_name = os.path.basename(input_folder)
        data_full_path = os.path.join('data_full', group_name)

        for dirpath, dirnames, filenames in os.walk(input_folder):
            for dirname in dirnames:
                base_name = dirname
                current_path_patches = os.path.join(dirpath, dirname)
                layers = [d for d in os.listdir(current_path_patches) if os.path.isdir(os.path.join(current_path_patches, d))]
                
                # Sort layers by numerical value to ensure correct order
                layers.sort(key=lambda x: int(re.search(r'layer_(\d+)', x).group(1)))

                full_image_accumulated = []

                for layer in layers:
                    layer_path = os.path.join(current_path_patches, layer)
                    patch_files = [f for f in os.listdir(layer_path) if f.endswith('.png')]
                    if not patch_files:
                        continue

                    patch_files.sort(key=lambda x: int(re.search(r'patch_(\d+)', x).group(1)))
                    patches = [os.path.join(layer_path, f) for f in patch_files]

                    dataset = PredCustomDataset(image_paths=patches, inference=True, denoise=False)
                    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                    preds = predict_image(loader, model, device)

                    original_image_path = os.path.join(data_full_path, f'{base_name}.tif')
                    original_image_raw = imread(original_image_path)
                    layer_height, layer_width = original_image_raw.shape[1:3]
                    shape_2d = (layer_height, layer_width)
                    #original_image = exposure.rescale_intensity(original_image_raw, in_range='image')
                    #original_image = exposure.equalize_hist(original_image)

                    full_image = reconstruct_image_from_patches(preds, shape_2d, patch_size=patch_size, overlap=overlap)
                    full_image_accumulated.append(full_image)
                    
                output_dir = os.path.join(output_base, group_name, base_name)
                os.makedirs(output_dir, exist_ok=True)
                

                with TiffWriter(os.path.join(output_dir, f'{base_name}_reconstructed_multi.tif')) as tif:
                    for layer_image in full_image_accumulated:
                        binary_image = np.where(layer_image, 255, 0).astype(np.uint8) 
                        tif.save(binary_image)
                
                print(f'Processed and saved multi-layer TIFF for {base_name} in {group_name}')


os.chdir('N:\\00_Exchange\\Alois')
input_folders = ['data_full/data_patch/FG12', 'data_full/data_patch/PBS', 'data_full/data_patch/NG004', 'data_full/data_patch/11C7']
output_base = 'data_processed'

process_folders(input_folders, output_base, model_path, patch_size=1024, overlap=0.3, batch_size=1)
print('Done')
