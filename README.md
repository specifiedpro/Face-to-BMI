# Face-to-BMI
BMI Prediction with ResNet50 and Auxiliary Features
This repository provides a PyTorch-based pipeline for predicting Body Mass Index (BMI) from images. It leverages a pretrained ResNet50 model, enhanced by an auxiliary sex feature in its regression head to refine predictions. Through extensive data augmentation and hyperparameter grid search, this project achieves a Pearson correlation of 0.665, surpassing the baseline of 0.65 established in prior work.

Key Features
Pretrained ResNet50 Backbone: Uses transfer learning to extract rich visual features from images.

Custom Regression Layer: Incorporates an additional sex feature for improved prediction accuracy.

Data Augmentation & Hyperparameter Tuning: Implements random transformations and a systematic search to optimize model performance.

Modular Codebase: Separates data preprocessing, model definition, and training scripts for clean, production-ready workflows.

Getting Started
Clone this repository and install dependencies via pip install -r requirements.txt.

Organize your data in the specified structure and update the paths in config.yaml.

Train the model by running python src/main.py to reproduce the reported results.

Use the contributions section of the README to see how you can adapt the code for different datasets or to add more features for BMI prediction. For more details, refer to the inline documentation and comments throughout the source files.
