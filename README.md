# Quantum Brain Tumor Detection

A hybrid quantum-classical deep learning model for binary brain tumor classification using MRI images. This project combines convolutional neural network feature extraction with a 4-qubit variational quantum circuit to explore quantum-enhanced medical image analysis.

## Overview

The architecture integrates EfficientNet-B0 as a classical backbone and a variational quantum layer implemented using PennyLane. Classical image features are compressed and embedded into a quantum circuit, processed through trainable entangling layers, and then passed to a final classification layer.

## Model Architecture

Classical Backbone:
- EfficientNet-B0 (modified for grayscale MRI input)
- Feature reduction layer before quantum embedding

## Research Publication

This project is associated with the following published research paper:

Title: Brain Tumour Detection Using Quantum Convolutional Neural Networks (QCNN)

Conference: IEEE Odisha International Conference on Information Technology (OCIT 2025)

Publisher: IEEE

DOI: https://doi.org/10.1109/OCIT66168.2025.11400476

Quantum Layer:
- 4 qubits
- AngleEmbedding for encoding classical features
- StronglyEntanglingLayers for parameterized quantum operations
- Implemented using PennyLane simulator

Output:
- Fully connected classification layer
- LogSoftmax activation for stable optimization

## Dataset

Dataset structure:

dataset/
 ├── yes/
 └── no/

Source: Publicly available Brain MRI dataset from Kaggle.

Images are resized to 224x224, converted to grayscale, and normalized before training.

## Training Configuration

- Optimizer: Adam
- Loss Function: Negative Log Likelihood (NLLLoss)
- Learning Rate Scheduler: ReduceLROnPlateau
- Train/Test Split: 80/20
- Batch Size: 16
- Epochs: 50

## Evaluation

Model performance was evaluated on a held-out 20% test split. Metrics include precision, recall, F1-score, and overall accuracy. Observed test accuracy is approximately 99%, with minor variation depending on data split and initialization.

## Inference

To run prediction:

python predict_qcnn.py

The interface provides tumor classification, confidence score, and Grad-CAM based localization.

## Technologies Used

- Python
- PyTorch
- Torchvision
- PennyLane
- NumPy
- OpenCV
- Gradio

## Project Context

This project was developed as an academic exploration of hybrid quantum neural networks for medical image classification.
