# BMI Prediction with ResNet50 

This repository implements a PyTorch-based pipeline for predicting Body Mass Index (BMI) from images. By leveraging a pretrained ResNet50 model and enhancing it with an auxiliary sex feature in a custom regression layer, this project achieves a Pearson correlation of **0.665**—exceeding the baseline of 0.65.

## Key Features
- **Pretrained ResNet50 Backbone**  
  Utilizes transfer learning to extract high-level visual features for effective BMI estimation.
- **Custom Regression Layer**  
  Integrates an additional sex feature to boost prediction accuracy.
- **Comprehensive Data Augmentation & Hyperparameter Tuning**  
  Applies various random transformations (e.g., resizing, cropping, color jittering) alongside a systematic grid search to optimize model performance.
- **Modular and Production-Ready Codebase**  
  Separates data preprocessing, model definition, training routines, and configuration for maintainability and scalability.


## Quick Start
1. **Clone the Repository & Install Dependencies:**
   ```bash
   git clone https://github.com/yourusername/BMI_Final_Proj.git
   cd BMI_Final_Proj
   pip install -r requirements.txt
Configure the Project:

Update the src/config.yaml file to reflect your data paths and preferred hyperparameters.

Run the Training Pipeline:

bash
Copy
python src/main.py
Results
Achieved a Pearson correlation of 0.665 on the test set, surpassing the published baseline of 0.65.

Detailed metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) are logged during training and validation.

Contributing
Contributions are welcome! Please consider opening an issue or submitting a pull request with improvements or additional features, such as:

Enhanced data preprocessing or augmentation techniques.

Alternative model architectures.

Extended evaluation metrics and visualizations.

License
This project is licensed under the MIT License.

Acknowledgements
This project builds upon insights from existing BMI prediction literature and leverages the power of PyTorch for model development and training.
