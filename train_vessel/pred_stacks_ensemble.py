import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader
from tifffile import imread, TiffWriter
from skimage import exposure
from utils_pred import PredCustomDataset, reconstruct_image_from_patches

#from resnet50_unet import UNetWithResnet50Encoder
from resnet152_attention_unet import RAUNet
from resnet152_unet import UNetWithResnet152Encoder
from resnet152_unet_SEB import UNetWithResnet152Encoder as UNetWithResnet152Encoder_SEB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Load models
name_model1 = 'tuned_resnet152_attention_unet_power_dice_dice'
name_model2 = 'resnet152_SEB_V5'
name_model3 = 'best_model_DRIVE_tuned152'

model1 = RAUNet(num_classes=1, num_channels=3, pretrained=True).to(device)
model1.load_state_dict(torch.load(f'output_NEW/{name_model1}.pth'))
model1.eval()

model2 = UNetWithResnet152Encoder_SEB(n_classes=1).to(device)
model2.load_state_dict(torch.load(f'output_NEW/{name_model2}.pth'))
model2.eval()

model3 = UNetWithResnet152Encoder(n_classes=1).to(device)
model3.load_state_dict(torch.load(f'output_dir/{name_model3}.pth'))
model3.eval()

models = [model1, model2, model3]

# Predict function
def predict_image(loader, models, device):
    """
    Predicts the segmentation masks for a batch of images using an ensemble of models.

    Args:
        loader (torch.utils.data.DataLoader): The data loader for the images.
        models (list): A list of models to use for prediction.
        device (torch.device): The device to perform the prediction on.

    Returns:
        list: A list of predicted segmentation masks for the input images.
    """
    for model in models:
        model.eval()
    preds = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            avg_pred_prob = 0
            
            for model in models:
                output = model(images)
                pred_prob = torch.sigmoid(output)
                avg_pred_prob += pred_prob / len(models)
            
            pred = (avg_pred_prob > 0.5).float()
            preds.extend(pred.detach().cpu().numpy())
    return preds

def process_folders(input_folders, output_base, models, patch_size=512, overlap=0.3, batch_size=8):
    """
    Process the input folders containing image patches and reconstruct multi-layer TIFFs.

    Args:
        input_folders (list): List of input folders containing image patches.
        output_base (str): Base directory for saving the output TIFFs.
        models (list): List of models for prediction.
        patch_size (int, optional): Size of the image patches. Defaults to 512.
        overlap (float, optional): Overlap between adjacent patches. Defaults to 0.3.
        batch_size (int, optional): Batch size for prediction. Defaults to 8.
    """
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
                    preds = predict_image(loader, models, device)

                    original_image_path = os.path.join(data_full_path, f'{base_name}.tif')
                    original_image_raw = imread(original_image_path)
                    layer_height, layer_width = original_image_raw.shape[1:3]
                    shape_2d = (layer_height, layer_width)

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

process_folders(input_folders, output_base, models, patch_size=1024, overlap=0.3, batch_size=1)
print('Done')
