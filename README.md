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
