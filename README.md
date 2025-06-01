# Mango Detection using YOLO

## Overview

This project implements a robust mango detection system using various versions of the YOLO (You Only Look Once) object detection algorithm. The system is designed to identify and classify mangoes in images, providing accurate bounding box detection for agricultural applications.

## Project Structure

```

├── Compare.ipynb          # Notebook for comparing different model versions
├── Models                 # Trained model weights
│   ├── mango_detector_v10.pt
│   ├── mango_detector_v12.pt
│   └── mango_detector_v8.pt
├── README.md
└── Training               # Training notebooks for different YOLO versions
    ├── v10.ipynb
    ├── v12.ipynb
    └── v8.ipynb
```

## Features

- **Multi-version YOLO Implementation**: Implements YOLOv8, YOLOv10, YOLOv11, and YOLOv12 for comparative analysis
- **GPU-Accelerated Training**: Optimized for training with GPU support
- **Comprehensive Metrics**: Evaluation metrics including mAP, precision, recall, and F1-score
- **Model Comparison**: Tools to compare inference speed and detection accuracy across models
- **Interactive Visualization**: Visual representation of detection results and performance metrics

## Dataset

The project uses [this](https://universe.roboflow.com/luigui-andre-cerna-grados-dpsrr/clasificacion-de-mangos) mango image dataset from Roboflow:
- **Source**: Luigui Andre Cerna Grados' "Clasificación-de-mangos" dataset
- **Version**: 16

## Installation

### Requirements
- Python 3.8+

### Setup

```bash
pip install ultralytics roboflow opencv-python matplotlib pandas seaborn numpy torch
```

## Usage

### Training a Model

1. Open the desired training notebook (e.g., `Training/v12.ipynb`)
2. Run the notebook cells to:
   - Download and preprocess the dataset
   - Train the model with defined parameters
   - Evaluate and save the model

### Comparing Models

1. Open `Compare.ipynb`
2. Upload the trained models from the Models directory
3. Run the comparison cells to:
   - Evaluate metrics across different model versions
   - Compare inference speed and detection accuracy
   - Visualize detection results side by side

### Inference

```python
from ultralytics import YOLO

# Load model
model = YOLO('Models/mango_detector_v12.pt')

# Run inference
results = model.predict(
    source='path/to/image.jpg',
    conf=0.5,
    save=True
)

# Process results
for r in results:
    boxes = r.boxes
    print(f"Found {len(boxes)} mangoes in the image")
```

## Model Performance

A comparison of the model versions:

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1-Score | Inference Time (s) | Training Time |
|-------|---------|--------------|-----------|--------|----------|----------|----------|
| YOLO v8 | 0.9087 | 0.6425 | 0.9052 | 0.8534 | 0.8785 | 13.3864 | 49 min |
| YOLO v10 | 0.8971 | 0.6178 | 0.8900 | 0.8305 | 0.8592 | 7.4955 | 55.5 min |
| YOLO v11 | 0.9194 | 0.6410 | 0.9341 | 0.8336 | 0.8810 | 7.0858 | 52.2 min |
| YOLO v12 | 0.9162 | 0.6412 | 0.9165 | 0.8291 | 0.8706 | 7.4938 | 69.3 min |

## Development Process

1. **Data Preprocessing**: Images are normalized and augmented to enhance training
2. **Model Training**: Models are trained with optimized hyperparameters
3. **Evaluation**: Comprehensive metrics are calculated to assess model performance
4. **Comparison**: Different YOLO versions are compared to identify the best-performing model

## License

This project is licensed under the terms of the MIT license.

## Acknowledgments

- Dataset provided by Luigui Andre Cerna Grados
- Built with Ultralytics YOLO implementation
- Developed using Google Colab for GPU acceleration
