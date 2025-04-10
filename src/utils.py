# utils.py

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr

def evaluate_model(model, dataloader, device):
    """
    Evaluate the ResNet50 model on the provided dataloader and return the ground truth and predicted BMI values.

    Args:
        model (torch.nn.Module): Trained ResNet50 model.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation data.
        device (str): Device to run evaluation on ('cuda' or 'cpu').

    Returns:
        np.array: Ground truth BMI values.
        np.array: Predicted BMI values.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y, sex in dataloader:
            # Move data to the specified device.
            X, y, sex = X.to(device), y.to(device), sex.to(device)
            # Forward pass: model expects image and sex feature.
            preds = model(X, sex)
            preds = preds.squeeze(1)  # Adjust output shape to [batch_size]
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    return np.array(all_targets), np.array(all_preds)

def compute_regression_metrics(y_true, y_pred):
    """
    Compute common regression metrics for BMI prediction.

    Args:
        y_true (np.array): True BMI values.
        y_pred (np.array): Predicted BMI values.

    Returns:
        dict: Dictionary containing MSE, RMSE, MAE, MAPE, Pearson correlation coefficient, and p-value.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    pearson_corr, p_value = pearsonr(y_true, y_pred)
    
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Pearson Correlation": pearson_corr,
        "Pearson p-value": p_value
    }
    
    return metrics
