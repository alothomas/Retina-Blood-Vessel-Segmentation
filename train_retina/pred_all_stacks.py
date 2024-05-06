import torch
import numpy as np
from PIL import Image
import tifffile as tiff
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
from torch.utils.data import Dataset
from glob import glob
import sys
sys.path.append('E:/Retina-Segmentation/retina_masking')


from models_retina.resnet50_unet import UNetWithResnet50Encoder
from utils import BinaryLovaszHingeLoss, DiceLoss, JaccardLoss, dice_coefficient, show_images_and_masks
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class PredictionDataset(Dataset):
    """
    Dataset class for prediction.

    Args:
        root_dir (str): Root directory of the dataset.
        augment (bool, optional): Whether to apply data augmentation. Defaults to False.
        denoise (bool, optional): Whether to apply denoising. Defaults to False.
    """

    def __init__(self, root_dir, augment=False, denoise=False):
        self.root_dir = root_dir
        self.denoise = denoise
        self.augment = augment
        self.image_files = glob(os.path.join(root_dir, '**', '*.png'), recursive=True)

        self.transform = A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if self.denoise:
            image = cv2.fastNlMeansDenoising(image, None, h=15, templateWindowSize=7, searchWindowSize=21)

        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = np.stack([image] * 3, axis=-1)
        transformed = self.transform(image=image)
        image = transformed['image']
        
        
        relative_path_parts = image_path.split(os.sep)[-3:] 
        condition, subject, filename = relative_path_parts
        layer_index = int(filename.split('_')[-1].split('.')[0]) 

        return image, condition, subject, layer_index

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_and_save_masks(model, dataset, original_dir_base, target_dir_base):
    """
    Process and save masks for a given model, dataset, original directory base, and target directory base.

    Args:
        model (torch.nn.Module): The model used for mask prediction.
        dataset (torch.utils.data.Dataset): The dataset containing images, conditions, subjects, and layer indices.
        original_dir_base (str): The base directory path where the original images are stored.
        target_dir_base (str): The base directory path where the predicted masks will be saved.
    """
    model.eval()
    for image, condition, subject, layer_index in dataset:
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mask_logits = model(image)
            pred_mask_prob = torch.sigmoid(pred_mask_logits)
            pred_mask = (pred_mask_prob > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255
        
        original_image_path = os.path.join(original_dir_base, condition, f'{subject}.tif')
        original_image = tiff.imread(original_image_path)
        
        if original_image.ndim == 2:
            original_image = original_image[np.newaxis, ...]
        
        original_layer = original_image[layer_index]
        pred_mask_rescaled = cv2.resize(pred_mask, original_layer.shape[::-1], interpolation=cv2.INTER_AREA)

        mask_tiff_path = os.path.join(target_dir_base, condition, f'{subject}_mask.tif')
        ensure_dir(mask_tiff_path)
        if os.path.exists(mask_tiff_path):
            mask_tiff = tiff.imread(mask_tiff_path)
            mask_tiff[layer_index] = pred_mask_rescaled
            tiff.imwrite(mask_tiff_path, mask_tiff, dtype=np.uint8)
        else:
            mask_tiff = np.zeros_like(original_image, dtype=np.uint8)
            mask_tiff[layer_index] = pred_mask_rescaled
            tiff.imwrite(mask_tiff_path, mask_tiff, dtype=np.uint8)

model = UNetWithResnet50Encoder(n_classes=1).to(device)
model.load_state_dict(torch.load('E:\\Retina-Segmentation\\retina_masking\\output_dir\\best_model_retina_fine_tuned.pth'))

os.chdir('N:\\00_Exchange\\Alois')


base_dir = 'data_retina_full/rescaled_retina'
conditions = ['FG12', 'PBS', 'NG004', '11C7']
for condition in conditions:
    dataset = PredictionDataset(os.path.join(base_dir, condition))
    process_and_save_masks(model, dataset, 'data_full', 'data_full/retina_masks')

print('Done!')

