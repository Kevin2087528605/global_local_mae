## GLMAE
## Introduction
GLMAE (Global Local Masked Autoencoder) is a self-supervised learning framework for $\Phi$-OTDR event classification, which effectively extracts spatiotemporal features using global and local masking strategies. The model demonstrates excellent performance, particularly on the BJTU-OTDR dataset.

  ![image](https://github.com/user-attachments/assets/bf34d882-dd3a-46ca-8570-5d147674e3c9)


## Installation
## Install Dependencies
To run this project, first install all the necessary dependencies. You can install them using the following command:

```
  pip install -r requirements.txt
```

## Download Pretrained Model
You need to download the pretrained model file vit-t-classifier-from_scratch_gl.pt via the Baidu Cloud Drive link below:

Download link: https://pan.baidu.com/s/1lULAcKOTGn5l-Zav9hQl-A?pwd=7fdo

Extraction code: 7fdo

Place the downloaded model file in your project directory for loading and use.

## Prepare Data
The dataset used in this project is sourced from https://github.com/BJTUSensor/Phi-OTDR_dataset_and_codes

Make sure you have correctly prepared the data according to the instructions in the link before proceeding with the experiments.

## Model Performance Comparison

We compare the performance of the $\phi$-GLMAE model with several state-of-the-art methods for $\Phi$-OTDR event classification, covering a wide range of machine learning and deep learning approaches. Below is the model performance on the BJTU-OTDR dataset:

![image](https://github.com/user-attachments/assets/8c31ac7e-2a7e-4d05-b5db-32054065b101)

## Running Tests
After completing the environment setup, data preparation, and downloading the pretrained model, you can run the test script to evaluate the model's performance using the following command:

```
python gl_mae_classifier_test.py
```
