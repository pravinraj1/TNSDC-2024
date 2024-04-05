MNIST Classification with Convolutional Neural Networks
This repository contains code for training a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using PyTorch. It also includes an example of using the trained model for inference on a single image.

Getting Started
To get a copy of this project up and running on your local machine for development and testing purposes, follow these steps:

Prerequisites
Make sure you have Python 3.x installed on your machine. You'll also need to install the following Python packages:

pip install torch torchvision matplotlib numpy
Installation
Clone the repository:
[git clone https://github.com/pavinraj1/mnist-cnn.git](https://github.com/pravinraj1/TNSDC-2024/blob/main/Handwritten_Digit_Recognition.ipynb)

Navigate to the cloned repository:

cd mnist-cnn
Training the Model
Run the training script to train the CNN model:

python train.py
Inference
After training, you can use the trained model for inference on a single image:


python inference.py
This will display the image along with the predicted class probabilities.

Built With
PyTorch - Deep learning framework used
Matplotlib - Plotting library
Authors
Your Name
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
This code is based on the MNIST classification tutorial provided in the PyTorch documentation.
