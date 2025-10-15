# Drowsiness Detection Project

This repository contains the implementation of two deep learning models for detecting drowsiness using eye state classification: a Custom CNN model and a MobileNet-based model. The models classify eye images as either "Open Eyes" or "Closed Eyes" to monitor driver alertness and prevent accidents caused by drowsiness.

## Project Overview

Drowsiness detection is a crucial task in driver safety systems. It helps in identifying fatigue signs early, potentially reducing traffic accidents. This project trains convolutional neural networks to detect whether eyes are open or closed from image data.

## Features

- Custom CNN architecture for eye state classification.
- Transfer learning with MobileNet for improved accuracy.
- Dataset preprocessing including grayscale conversion and resizing.
- Training and validation accuracy/loss visualization.
- Model evaluation on a separate test set.
- Prediction on new images.

## Files in the Repository

- `Drowsiness-Detection_Custom_Model.ipynb`: Jupyter notebook with the custom CNN model implementation, training, and evaluation.
- `Drowsiness-Detection_MobilNet.ipynb`: Notebook with MobileNet model implementation and training.
- `Final.pdf`: Project report detailing methodology, experiments, results, and conclusions.

## Installation

1. Clone the repository:


cd drowsiness-detection


2. Install required packages:

pip install -r requirements.txt


Required packages include TensorFlow, OpenCV, matplotlib, numpy, pandas, and others as used in the notebooks.

## Usage

### Training the Custom Model

Open `Drowsiness-Detection_Custom_Model.ipynb` and run the cells to preprocess data, build the CNN model, train it, and evaluate performance.

### Training the MobileNet Model

Similarly, open `Drowsiness-Detection_MobilNet.ipynb` to train and evaluate the MobileNet based model.

### Testing with New Images

You can test the trained models by loading new images, preprocessing them the same way as training data, and running predictions.


## References

- Dataset source details.
- Relevant papers and resources on drowsiness detection and models.
