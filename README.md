**Malaria Detection Using Convolutional Neural Networks**
This repository contains the implementation of a deep learning model to detect malaria from cell images using 
TensorFlow. The model is built  using convolutional neural networks (CNNs) and is trained, validated, and 
tested using the Malaria dataset from TensorFlow Datasets.

**Project Overview**
The goal of this project is to build a CNN that can accurately distinguish between parasitized and unparasitized 
blood cell images. This involves loading the data, preprocessing it, visualizing samples, constructing 
and training the CNN, and evaluating its performance.

**Prerequisites**
To run this project, you will need:
- Python 3.x
- TensorFlow 2.x
- TensorFlow Datasets
- NumPy
- Matplotlib
- Install the required packages using the command below (bash):
- pip install tensorflow tensorflow-datasets numpy matplotlib
Clone the repository to your local environment

**Running the File**
Once cloning is complete:
Open and run the Jupyter notebook using the command below (bash):
jupyter notebook malaria_detection.ipynb

**Features**
- Data Loading and Preparation: Load the malaria dataset and organize it into training, validation, and test datasets.
- Image Preprocessing: Resize and rescale the images to ensure uniformity before feeding them into the model.
- CNN Model Building: Construct a CNN using layers like Conv2D, MaxPool2D, Dense, and BatchNormalization.
- Training and Validation: Train the model and validate it to monitor performance and avoid overfitting.
- Model Evaluation: Evaluate the model's accuracy and loss on the test dataset.
- Visualization: Visualize the predictions alongside the actual labels to provide a qualitative assessment of the model's performance.

**Model Architecture**
The CNN architecture is defined as follows:
- Input Layer: Adjusts input image size to 224x224 pixels.
- Conv2D and MaxPool2D layers for feature extraction.
- BatchNormalization layers to stabilize and speed up training.
- Flatten layer to convert 2D features to 1D.
- Dense layers for classification, with relu and sigmoid activation for binary classification.

**Dataset**
The dataset used is malaria from TensorFlow Datasets, which includes labeled images of parasitized and unparasitized cells derived from blood smear slides.

**Visualizations and Results**
Model Loss and Accuracy: Visualizations for training and validation loss and accuracy.
Sample Predictions: Display predictions for a subset of test images to visually assess the model's performance.
