# Retina-Blood-Vessel-Segmentation


## Overview
This Retina Project utilizes advanced deep learning techniques for the segmentation of blood vessels and retina areas in confocal microscopy images of mouse retinas.

## Features
- Blood vessel segmentation using custom ResNet-UNet architectures.
- Retina area segmentation with high accuracy.
- Extensive use of data augmentation to enhance model robustness.
- Implementation of Attention Residual UNets for precise feature extraction.

The project includes several models:

    ResNet152-UNet SEB: Squeeze-and-Excitation blocks integrated into a UNet architecture with a ResNet-152 backbone.
    RA-UNet: A hybrid deep attention-aware network designed for precise segmentation tasks.
    Standard ResNet152-UNet: A robust segmentation model suitable for various medical imaging tasks.

For more details on model architectures and configurations, check the models directory.

## Repository Structure
- `Analysis/`: Analysis tools and scripts for post-processing.
- `Preprocessing/`: Scripts for preprocessing.
- `train_retina/`: Training and prediction scripts for retina segmentation.
- `train_vessels/`: Training and prediction scripts for blood vessel segmentation.

## Usage

Adapt the scripts with your folder structure.

### Segmentation

- Preprocessing: Extract layers of multi layer czi files with `layer_extraction_3D.py`. Create tiles using `tiling_3D.py`.
- Predict blood vessels using `pred_stacks_ensemble.py`, choose the models to use.
- Predict retinas using `pred_all_stacks.py` (choose corresponding model). Rescale all retinas using `rescale_all_stacks.py` before prediction.

- Training: You can train your own models using the corresponding training files on your own dataset.


### Analysis

All scripts related to analysis of the retinas are in the Analysis folder.

- Filtering of perfused areas can be done using `filtering_all.py`.
- For Skeleton analysis, extract the layers of files using `skeleton_IJ.py`. Skeletonize with ImageJ Fiji script `macro_skeleton.ijm`, adapt the script to the folder structure.
- Align the layers using the first script `align_layers.py`. Then you can explore the data with the notebooks such as `area_fraction.ipynb`. Only run afterwards the other data exploration scripts.
