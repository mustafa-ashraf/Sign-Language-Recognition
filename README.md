# Sign Language Digits Recognition

![Sign Language Digits Recognition](https://www.salihbout.com/img/posts/sign-cnn/preview-sign.png)

This project focuses on recognizing sign language digits using different neural network architectures.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

Sign language digit recognition is a challenging computer vision task that involves recognizing hand gestures representing numbers in sign language. In this project, we explore the use of various neural network architectures to achieve accurate recognition of sign language digits.

## Dataset

The dataset used in this project contains sign language digit images. The dataset is divided into grayscale and RGB images. It consists of images and corresponding labels for each sign language digit.

## Getting Started

Follow these instructions to get the project up and running on your local machine.

## Prerequisites

Make sure you have the following libraries and frameworks installed:
- Python
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow/Keras

## Data Preprocessing

Data preprocessing involves the following steps:

1. **Loading the dataset images**: This step includes reading the sign language digit images from the dataset.
2. **Converting RGB images to grayscale**: If using RGB images, this process converts them to grayscale for consistent processing.
3. **Normalizing the pixel values of the images**: Ensuring that pixel values are within a standardized range, typically between 0 and 1.

## Model Architectures

The project utilizes three different neural network architectures:

1. **First Neural Network (ANN)**: A simple feedforward neural network for digit recognition.
2. **Second Neural Network (ANN)**: Another feedforward neural network, allowing for architecture comparison.
3. **Convolutional Neural Network (CNN)**: A convolutional neural network specially designed for image recognition tasks.

## Training and Evaluation

The neural network models are trained and evaluated using k-fold cross-validation. This involves:

1. **Data Splitting**: The dataset is divided into training and test sets for each fold.
2. **Training**: The models are trained using the training data, and logs are generated during training.
3. **Evaluation**: The models are evaluated using the test data, and performance metrics are recorded.

## Results

The project compares the performance of the different model architectures, including:

- Training and test accuracy for each model.
- Classification reports for each model, providing more detailed insights into model performance.

## Conclusion

The project concludes by comparing the performance of the different models and recommending the best-performing architecture for recognizing sign language digits.


