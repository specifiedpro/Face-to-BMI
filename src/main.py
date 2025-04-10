# src/main.py
import yaml
import torch
import numpy as np
from src.data_prep import load_and_process_data, split_dataset, build_img_list
from src.dataset import CustomImageDataset
from src.model import ResNet50
from src.train import train_epoch, validate_epoch, test_model, hyperparameter_grid_search
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_config(config_path='src/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # Data parameters
    csv_path = config['data']['csv_path']
    image_base_path = config['data']['image_base_path']

    df = load_and_process_data(csv_path, image_base_path)
    df_train, df_valid, df_test = split_dataset(df)

    train_img_lst = build_img_list(df_train, image_base_path)
    valid_img_lst = build_img_list(df_valid, image_base_path)
    test_img_lst  = build_img_list(df_test, image_base_path)

    # Data transformations (define your train and valid transforms)
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    # Create datasets and dataloaders
    train_data = CustomImageDataset(train_img_lst, transform=train_transforms)
    valid_data = CustomImageDataset(valid_img_lst, transform=valid_transforms)
    test_data = CustomImageDataset(test_img_lst, transform=valid_transforms)

    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Run hyperparameter grid search (or a single training run)
    optimizers = config['training']['optimizer_choices']
    learning_rates = config['training']['learning_rates']
    grid_hist = hyperparameter_grid_search(ResNet50, train_loader, valid_loader, device, optimizers, learning_rates)
    print("Grid search results:", grid_hist)

    # Optionally, load the best model and run test evaluation.
    # ...

if __name__ == '__main__':
    main()
