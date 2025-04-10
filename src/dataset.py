# src/dataset.py
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, img_list, transform=None):
        """
        img_list: List of tuples (image_path, bmi, sex)
        transform: Image transformations to apply
        """
        self.img_list = img_list  
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path, bmi, sex = self.img_list[idx]
        # Open the image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Convert bmi and sex to tensors
        bmi = torch.tensor(bmi, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.float32)
        return image, bmi, sex
