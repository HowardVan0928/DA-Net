# DA-Net: Duirection Aware Network for Object Detection in Adverse Environment

DA-Net is an enhanced network that incorporates advanced architectural modifications to improve object detection performance, particularly for challenging scenarios.

## Overview

DA-Net introduces Directional Enhancement Blocks (DEB) and Local Directional Detection (LDD) blocks to enhance feature extraction and improve detection accuracy. The model maintains a lightweight architecture while providing superior performance compared to standard YOLO11 baseline.

## Key Features

Learnable Direction Decomposition (LDD) Convolution: Introduces LDD convolution to explicitly model directional information during early-stage downsampling. By decomposing features along multiple orientations, LDD preserves fragile boundary and structural cues that are commonly lost in standard convolutional operations.

Directional Enhancement Block (DEB): Proposes a Directional Enhancement Block to hierarchically refine and aggregate directional features. DEB strengthens orientation-aware representations and improves robustness to visual degradation caused by haze and low-contrast conditions.

NexusFPN: Direction-Aware Multi-Path Feature Pyramid Network: Presents NexusFPN, a bidirectional and direction-aware feature pyramid network that enhances multi-scale feature fusion while maintaining structural coherence across scales, particularly benefiting small and densely distributed objects.

## Usage

### Installation

    #Install Pytorch
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    
    # Clone the repository
    git clone https://github.com/yourusername/DA-Net.git
    cd DA-Net

    # Install dependencies
    pip install -e .

### Training
    # Train a DA-Net model on your custom dataset
    yolo detect train model=yolo11-da.yaml data=your_dataset.yaml epochs=100 imgsz=640

### Validation
    # Validate a DA-Net model on your dataset
    yolo detect val model=yolo11-da.pt data=your_dataset.yaml

### Inference
    # Run inference with a trained DA-Net model
    yolo detect predict model=yolo11-da.pt source=path/to/images

### Weights
The trained path on HazyDet is https://pan.baidu.com/s/1_t_gPLWvYHmUmdhrk7FH4Q?pwd=tdvp
