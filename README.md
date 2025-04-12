# SuperPoint and SuperGlue Feature Matching

![Computer Vision](https://img.shields.io/badge/Computer%20Vision-blue?style=plastic)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-blue?style=plastic)
![Feature Matching](https://img.shields.io/badge/Feature%20Matching-blue?style=plastic)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.5-blue?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.0-blue?style=plastic)
![Torchvision](https://img.shields.io/badge/Torchvision-0.15.0-blue?style=plastic)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.2-blue?style=plastic)
![Numpy](https://img.shields.io/badge/Numpy-1.24.4-blue?style=plastic)
![Pandas](https://img.shields.io/badge/Pandas-2.0.3-blue?style=plastic)
![Pathlib](https://img.shields.io/badge/Pathlib-supported-blue?style=plastic)
![Build_Status](https://img.shields.io/badge/build-passing-brightgreen)
![Open_Issues](https://img.shields.io/badge/Issues-0-orange?style=plastic)

<p align="center">
  <img src="How it works.png" alt="Coronavirus" width="500"/>
</p>

## Introduction
This repository provides a comprehensive pipeline for **feature matching** using **SuperPoint** and **SuperGlue** models. The project focuses on matching aerial and satellite imagery using these advanced computer vision models. The main goal is to identify and match corresponding regions from two images (e.g., a drone image and a satellite tile) using the power of deep learning. The results can then be applied to tasks such as **GPS-based localization** and **flight path reconstruction**.

## Table of Contents
1. [Overview](#overview)  
2. [Dependencies Installation](#dependencies-installation)  
3. [Environment Setup](#environment-setup)  
4. [Extract Frames from Drone Video](#extract-frames-from-drone-video)  
6. [Display Sample Satellite Tile](#display-sample-satellite-tile)  
7. [Image Preprocessing](#image-preprocessing)  
8. [Initialize SuperPoint Model (v1 Weights)](#initialize-superpoint-model-v1-weights)  
9. [Extract Keypoints and Descriptors](#extract-keypoints-and-descriptors)  
10. [Visualize Sample Keypoints](#visualize-sample-keypoints)  
11. [Initialize SuperGlue Model (Outdoor Weights)](#initialize-superglue-model-outdoor-weights)  
12. [Perform Feature Matching](#perform-feature-matching)  
13. [Extract GPS from Satellite Tiles](#extract-gps-from-satellite-tiles)  
14. [Estimate GPS for Drone Frames](#estimate-gps-for-drone-frames)  
15. [Save Model and Output](#save-model-and-output)  
16. [Conclusion](#conclusion)  
17. [License](#license)

## Overview
This project builds a vision-based geolocation system that matches aerial drone frames to satellite imagery. It uses:
- **SuperPoint** for keypoint detection and description.
- **SuperGlue** for deep matching between image pairs.
- Satellite tile metadata to infer frame-level GPS positions.

## Dependencies Installation
Install required Python packages:
```bash
pip install opencv-python torch torchvision numpy pandas matplotlib tqdm
```
## Environment Setup
- Set up the necessary environment by importing required libraries and mounting Google Drive:
```bash
import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

## Extract Frames from Drone Video
- Extract frames from the drone video using OpenCV:
```bash
video_path = '/content/drive/MyDrive/DroneData/drone_video.mp4'
output_dir = '/content/drive/MyDrive/DroneData/frames'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
```
<p align="center">
  <img src="How it works.png" alt="Coronavirus" width="500"/>
</p>

## Display Sample Satellite Tile
- Display a sample satellite tile for visualization:
<p align="center">
  <img src="How it works.png" alt="Coronavirus" width="500"/>
</p>

## Image Preprocessing
- Preprocess images by converting to grayscale, resizing, and normalizing:
```bash
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (640, 480))
    image_normalized = image_resized / 255.0
    return image_normalized
```
<p align="center">
  <img src="How it works.png" alt="Coronavirus" width="500"/>
</p>

## Display Satellite tiles and frames after preprocessing
<p align="center">
  <img src="How it works.png" alt="Coronavirus" width="500"/>
</p>

## Initialize SuperPoint Model (v1 Weights)
- Initialize the SuperPoint model with pre-trained weights:
```bash
from models.superpoint import SuperPoint

superpoint_config = {
    'nms_radius': 4,
    'keypoint_threshold': 0.005,
    'max_keypoints': 1024
}

superpoint = SuperPoint(superpoint_config)

```
## Extract Keypoints and Descriptors
- Extract keypoints and descriptors from an image using SuperPoint:
```bash
image_tensor = torch.from_numpy(preprocessed_image).float().unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    outputs = superpoint({'image': image_tensor})
keypoints = outputs['keypoints'][0].cpu().numpy()
descriptors = outputs['descriptors'][0].cpu().numpy()
```
## Visualize Sample Keypoints
<p align="center">
  <img src="How it works.png" alt="Coronavirus" width="500"/>
</p>

## Initialize SuperGlue Model (Outdoor Weights)
- Initialize the SuperGlue model with outdoor pre-trained weights:
```bash
from models.superglue import SuperGlue

superglue_config = {
    'weights': 'outdoor',
    'sinkhorn_iterations': 20,
    'match_threshold': 0.2
}

superglue = SuperGlue(superglue_config)
```
## Perform Feature Matching
- Perform feature matching between two sets of keypoints and descriptors:
```bash
# Assume descriptors1, keypoints1 from image1 and descriptors2, keypoints2 from image2
data = {
    'keypoints0': torch.from_numpy(keypoints1).unsqueeze(0),
    'keypoints1': torch.from_numpy(keypoints2).unsqueeze(0),
    'descriptors0': torch.from_numpy(descriptors1).unsqueeze(0),
    'descriptors1': torch.from_numpy(descriptors2).unsqueeze(0),
    'image0': torch.from_numpy(image1).unsqueeze(0).unsqueeze(0),
    'image1': torch.from_numpy(image2).unsqueeze(0).unsqueeze(0)
}

with torch.no_grad():
    matches = superglue(data)

```
## Extract GPS from Satellite Tiles
- Extract GPS coordinates associated with each satellite tile:
```bash
import json

with open('/content/drive/MyDrive/SatelliteData/tiles_metadata.json', 'r') as f:
    tiles_metadata = json.load(f)

# Example: tiles_metadata = {'tile_001.jpg': {'latitude': 30.0444, 'longitude': 31.2357}, ...}
```
## Estimate GPS for Drone Frames
- Estimate GPS coordinates for drone frames based on matching with satellite tiles:
```bash
# Assuming best_match_tile is the filename of the best matching tile
gps_coordinates = tiles_metadata.get(best_match_tile, {'latitude': None, 'longitude': None})
```
## Save Model and Output
- Save the matching results and estimated GPS coordinates:
```bash
results = {
    'frame': 'frame_0000.jpg',
    'matched_tile': best_match_tile,
    'gps': gps_coordinates
}

with open('/content/drive/MyDrive/Results/matching_results.json', 'w') as f:
    json.dump(results, f, indent=4)
```
## Conclusion
This project demonstrates a pipeline for matching drone video frames with satellite tiles using SuperPoint and SuperGlue models. By extracting and matching features, it estimates the GPS coordinates of drone frames, facilitating geospatial analysis and navigation tasks.

## License
- This project is licensed under the MIT License - see the LICENSE file for details.
- Feel free to adjust any details to better fit your specific implementation or project structure.
