# Handwritten Digit Recognition using Convolutional Neural Networks

This repository contains code for training and evaluating a Convolutional Neural Network (CNN) model to recognize handwritten digits using the MNIST dataset.

## Introduction

Handwritten digit recognition is a classic problem in the field of computer vision and machine learning. The goal is to develop a model that can accurately classify images of handwritten digits into their corresponding numerical labels (0 to 9).

In this project, we utilize the MNIST dataset, which consists of grayscale images of handwritten digits, each of size 28x28 pixels. We employ a CNN architecture to learn discriminative features from the images and classify them into the correct digit classes.

## Setup

To run the code in this repository, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/pravinraj1/TNSDC-2024.git
    ```

2. Install the required dependencies. You can use the provided `requirements.txt` file to install the dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook or Python script to train the model, evaluate its performance, and make predictions.

## Contents

- `train_model.ipynb`: Jupyter notebook containing the code for training the CNN model on the MNIST dataset.
- `predict.ipynb`: Jupyter notebook demonstrating how to make predictions using the trained model.
- `model.py`: Python script defining the architecture of the CNN model.
- `utils.py`: Python script containing utility functions for data loading and preprocessing.
- `requirements.txt`: File specifying the required Python packages and their versions.

## Usage

- `train_model.ipynb`: Open this notebook in Jupyter and execute each cell to train the CNN model. You can adjust hyperparameters, such as learning rate and batch size, as needed.
- `predict.ipynb`: Use this notebook to make predictions on new handwritten digit images using the trained model. You can upload your own images or use sample images provided in the repository.

## Results

After training the model, you can expect to achieve an accuracy of over 95% on the test set, indicating the model's ability to accurately classify handwritten digits.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The code in this repository is inspired by various tutorials and resources on deep learning and convolutional neural networks.
- Special thanks to the creators of the MNIST dataset for providing a valuable resource for benchmarking machine learning models.
