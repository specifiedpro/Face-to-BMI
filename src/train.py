# src/train.py
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
import os

def train_epoch(dataloader, model, loss_function, optimizer, device, log_interval=32):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y, sex) in enumerate(dataloader):
        X, y, sex = X.to(device), y.to(device), sex.to(device)
        pred = model(X, sex)
        loss = loss_function(pred.squeeze(1), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % log_interval == 0:
            current = batch * len(X)
            print(f"Train loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

def validate_epoch(dataloader, model, loss_function, device):
    model.eval()
    total_loss, total_mae, total_samples = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y, sex in dataloader:
            X, y, sex = X.to(device), y.to(device), sex.to(device)
            pred = model(X, sex)
            loss = loss_function(pred.squeeze(1), y)
            mae_loss = nn.L1Loss()(pred.squeeze(1), y)
            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            total_mae += mae_loss.item() * batch_size
            total_samples += batch_size
    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    print(f"Validation Error:\n MSE Loss: {avg_loss:>8f} \n MAE Loss: {avg_mae:>8f}\n")
    return avg_loss, avg_mae

def test_model(dataloader, model, loss_function, device):
    y_true, y_pred = [], []
    total_loss, total_samples = 0.0, 0
    model.eval()
    with torch.no_grad():
        for X, y, sex in dataloader:
            X, y, sex = X.to(device), y.to(device), sex.to(device)
            pred = model(X, sex)
            y_pred.extend(pred.squeeze(1).cpu().numpy())
            y_true.extend(y.cpu().numpy())
            batch_size = X.size(0)
            total_loss += loss_function(pred.squeeze(1), y).item() * batch_size
            total_samples += batch_size
    avg_loss = total_loss / total_samples
    mae = np.mean(np.abs(np.array(y_pred) - np.array(y_true)))
    print(f"Test Error:\n Avg Loss: {avg_loss:>8f}\n Avg MAE: {mae:>8f}")
    return avg_loss, mae, y_true, y_pred

def hyperparameter_grid_search(model_class, train_loader, valid_loader, device, optimizers, learning_rates):
    grid_hist = {}
    for opt_name in optimizers:
        grid_hist[opt_name] = {}
        for lr_idx, lr in enumerate(learning_rates):
            # Skip extra learning rates if needed.
            if lr_idx >= 4:
                print('Skipping lr of', lr)
                continue
            print(f'============== Training with {opt_name} at lr = {lr} ==============')
            model = model_class().to(device)
            if opt_name == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            elif opt_name == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
            criterion = nn.MSELoss()
            max_epochs = 250
            current_min_loss = np.inf
            earlyStop_cnt = 0
            for epoch in range(max_epochs):
                print(f"Epoch {epoch+1}\n-------------------------------")
                train_epoch(train_loader, model, criterion, optimizer, device)
                eval_loss, _ = validate_epoch(valid_loader, model, criterion, device)
                # Save best model based on validation loss
                if eval_loss < current_min_loss:
                    best_model_path = f'best_{opt_name}_{lr_idx}.pth'
                    torch.save(model.state_dict(), best_model_path)
                    print('Best model saved!', eval_loss)
                    current_min_loss = eval_loss
                    earlyStop_cnt = 0
                else:
                    earlyStop_cnt += 1
                if earlyStop_cnt >= 20:
                    print('Early stopping!')
                    break
            grid_hist[opt_name][lr_idx] = current_min_loss
    return grid_hist
