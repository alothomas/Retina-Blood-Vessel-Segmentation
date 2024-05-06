import os
import glob
import cv2
import numpy as np
import random
from PIL import Image
import tifffile

##############################################
###### BLOOD VESSEL DENSITY CALCULATION ######


def find_image_paths(base_dir):
    image_paths = {'FG12': {}, 'PBS': {}, 'NG004': {}, '11C7': {}}
    
    # Directly replace 'data_processed' with '' in base_dir to adjust the path
    base_dir_adjusted = base_dir.replace('data_processed', '')  
    retina_mask_base_dir = os.path.join(base_dir_adjusted, 'retina_masks')

    for treatment in ['FG12', 'PBS', 'NG004', '11C7']:
        treatment_dir = os.path.join(base_dir, treatment)

        subjects = [d for d in os.listdir(treatment_dir) if os.path.isdir(os.path.join(treatment_dir, d))]
        
        for subject in subjects:
            subject_dir = os.path.join(treatment_dir, subject)

            blood_vessels_glob = glob.glob(os.path.join(subject_dir, '*reconstructed.tif'))
            #blood_vessels_glob = glob.glob(os.path.join(subject_dir, '*reconstructed.tif'))

            
            retina_mask_dir = os.path.join(retina_mask_base_dir, treatment)
            #retina_surface_glob = glob.glob(os.path.join(retina_mask_dir, f'{subject}_retina_mask.tif'))
            retina_surface_glob = glob.glob(os.path.join(retina_mask_dir, f'{subject}_retina_mask.tif'))

            
            if not blood_vessels_glob or not retina_surface_glob:
                print(f"Missing files for {subject} in {treatment}.")
                continue 
            
            blood_vessels_path = blood_vessels_glob[0]
            retina_surface_path = retina_surface_glob[0]
            
            eye = 'OD' if 'OD' in subject else 'OE'
            if subject not in image_paths[treatment]:
                image_paths[treatment][subject] = {}
                
            image_paths[treatment][subject][eye] = {
                'blood_vessels': blood_vessels_path,
                'retina_surface': retina_surface_path
            }
    
    return image_paths



from PIL import Image

def calculate_vessel_density(blood_vessels_path, retina_surface_path):
    blood_vessels_img = tifffile.imread(blood_vessels_path)
    
    #_, blood_vessels_img = cv2.threshold(blood_vessels_img, 127, 255, cv2.THRESH_BINARY)

    retina_surface_img = tifffile.imread(retina_surface_path)   
    #_, retina_surface_img = cv2.threshold(retina_surface_img, 127, 255, cv2.THRESH_BINARY)
    
    sum_retina_surface_img = np.sum(retina_surface_img > 0)
    sum_vessel_img = np.sum(blood_vessels_img > 0)

    if sum_retina_surface_img == 0:
        return 0
    if sum_vessel_img == 0:
        return 0
    else:
        vessels_density = sum_vessel_img / sum_retina_surface_img
        return vessels_density
    


def aggregate_vessel_densities(image_paths):
    """
    Aggregate the vessel densities for each subject and eye based on the provided image paths.
    
    Parameters:
    - image_paths: A dictionary containing the paths to blood vessel and retina surface images for each subject.
    
    Returns:
    - A dictionary with the aggregated vessel densities.
    """
    results = {'FG12': {}, 'PBS': {}, 'NG004': {}, '11C7': {}}
    for treatment in image_paths.keys():
        for subject, eyes in image_paths[treatment].items():
            if subject not in results[treatment]:
                results[treatment][subject] = {}
            for eye, paths in eyes.items():
                vessel_density = calculate_vessel_density(paths['blood_vessels'], paths['retina_surface'])
                results[treatment][subject][eye] = vessel_density
    return results

