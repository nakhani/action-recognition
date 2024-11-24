# Action Recognition with MoSIFT

This repository contains an implementation of action recognition using the MoSIFT (Motion SIFT) method. It includes two main Python files that handle data preprocessing and model training, utilizing libraries such as PyTorch, OpenCV, and NumPy.

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
- [Results](#results)


## Introduction

Action recognition is a crucial task in computer vision, with applications ranging from video surveillance to human-computer interaction. This repository demonstrates the use of the MoSIFT method to identify actions in video sequences. The MoSIFT algorithm combines motion information with SIFT keypoints to enhance action recognition accuracy.

## Repository Structure

- **neuralnet.py**: Contains the dataset processing and model training scripts.
- **objs.py**: Handles feature extraction from video files using MoSIFT and optical flow.

## Setup and Installation

To run this project, you need to have Python installed along with the required libraries. Follow these steps to set up the environment:

1. **Clone this repository**:
    ```bash
    git clone https://github.com/nakhani/action-recognition.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd action-recognition
    ```

3. **Create and activate a virtual environment (optional but recommended)**:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

4. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preprocessing

Before training the model, you need to preprocess the video data and extract features using the MoSIFT algorithm.

1. **Feature Extraction**:
   Use `objs.py` to extract MoSIFT features from your video files.
   ```bash
   python objs.py --input_path path/to/videos --output_path path/to/output

2. **Data Splitting**:
   Ensure that your data is split into training and validation sets. The provided script `neuralnet.py`  includes functionality to split the dataset.

## Model Training
Train the action recognition model using the preprocessed data.


1. **Run the Training Script**:
   ```bash
   python neuralnet.py
   ```
This script will train the model and display the training and validation accuracy. The trained model will be saved in the specified directory.

## Result 
The trained model achieves notable accuracy in recognizing actions from video sequences. Detailed results and accuracy metrics are printed during the training process.
