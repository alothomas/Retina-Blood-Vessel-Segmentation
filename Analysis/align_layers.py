import os
import glob
import cv2
import numpy as np
import random
from adjustText import adjust_text
import plotly.express as px
from tifffile import TiffFile
import os
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(42)

# Set the base directory
base_dir = r"N:\00_Exchange\Alois\data\data_processed"
os.chdir(r"N:\\00_Exchange\\Alois")



def compute_area_fraction(data_folder):
    """
    Computes the area fraction of blood vessels in retinal images.

    Args:
        data_folder (str): The path to the folder containing the data.

    Returns:
        pandas.DataFrame: A DataFrame containing the computed area fraction results.

    """
    results = []
    
    treatments = ['FG12', 'PBS', '11C7', 'NG004']
    
    for treatment in treatments:
        vessel_path = os.path.join(data_folder, 'data_processed', treatment)
        mask_path = os.path.join(data_folder, 'retina_masks', treatment)
       
        
        for subject in os.listdir(vessel_path):
            print(f'Processing subject: {subject}')

            subject_folder = os.path.join(vessel_path, subject)
            vessel_file = os.path.join(subject_folder, f'{subject}_reconstructed_multi.tif')
            mask_file = os.path.join(mask_path, f'{subject}_mask.tif')

            
            if os.path.exists(vessel_file) and os.path.exists(mask_file):

                layers_vessel = []
                with TiffFile(vessel_file) as tif:
                    for page in tif.pages:
                        layer = page.asarray()
                        layers_vessel.append(layer)
                vessel_img = np.stack(layers_vessel)



                mask_img = tiff.imread(mask_file)

                num_layers = min(vessel_img.shape[0], mask_img.shape[0])
                
                for layer in range(num_layers):
                    vessel_layer = vessel_img[layer]
                    mask_layer = mask_img[layer]
                    
                    retina_area = (mask_layer > 0).sum()
                    if retina_area > 0:
                        vessel_area = (vessel_layer > 0).sum()
                        area_fraction = vessel_area / retina_area
                    else:
                        area_fraction = 0
                    
                    eye = 'OD' if 'OD' in subject else 'OE'
                    results.append({
                        'Treatment': treatment,
                        'SubjectID': subject,
                        'Eye': 'Right' if eye == 'OD' else 'Left',
                        'LayerNumber': layer,
                        'AreaFraction': area_fraction,
                        'VesselArea': retina_area
                    })
    
    results_df = pd.DataFrame(results)
    return results_df


def align_layers(group):
    """
    Aligns the layers in a group based on the maximum area fraction layer.

    Parameters:
    - group: pandas DataFrame
        The group of layers to be aligned.

    Returns:
    - group: pandas DataFrame
        The aligned group of layers.
    """
    max_area_layer = group['AreaFraction'].idxmax()
    shift = group.loc[max_area_layer, 'LayerNumber']
    group['AlignedLayerNumber'] = group['LayerNumber'] - shift
    return group



# Path to the data_full folder
data_folder = 'data_full'
df = compute_area_fraction(data_folder)

df['SubjectID'] = df['SubjectID'].str.extract(r'Akita(\d+)')
df['Eye'] = df['Eye'].replace({'Left': 'OE', 'Right': 'OD'})


# Export CSV with total retina area
df.to_csv('total_area.csv', index=False)


# Align layers
df_aligned = df.groupby(['SubjectID', 'Eye'], as_index=False).apply(align_layers)
df_aligned.reset_index(drop=True, inplace=True)
df_aligned.sort_values(by=['SubjectID', 'Eye', 'AlignedLayerNumber'], inplace=True)


df_aligned.to_csv('area_fraction_aligned_layers.csv', index=False)

# Export CSV with alignment information
df_export = df_aligned[['SubjectID', 'Eye', 'Treatment', 'LayerNumber', 'AlignedLayerNumber']]
df_export.to_csv('alignment_layers.csv', index=False)


# Export CSV with area fraction
df = df_aligned
df = df.drop(columns=['LayerNumber'])
df = df.rename(columns={'AlignedLayerNumber': 'LayerNumber'})
df.to_csv('area_fraction.csv', index=False)


print('Done')