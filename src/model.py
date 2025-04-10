# src/model.py
import torch
import torch.nn as nn
import torchvision

class ResNet50(nn.Module):
    def __init__(self, load_weights=True, freeze_hidden_layers=False):
        super(ResNet50, self).__init__()
        # Load pretrained ResNet50 model.
        self.base_model = torchvision.models.resnet50(pretrained=load_weights)
        
        # Remove the original fully connected layer.
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        
        # New regressor layer (combining image features and extra sex feature)
        self.fc_reg = nn.Linear(num_ftrs + 1, 1)
        
        if freeze_hidden_layers:
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.fc_reg.parameters():
                param.requires_grad = True

    def forward(self, x, sex):
        # Extract features from image.
        features = self.base_model(x)
        if len(sex.shape) == 1:
            sex = sex.unsqueeze(1)
        # Concatenate features with extra feature.
        combined = torch.cat((features, sex), dim=1)
        output = self.fc_reg(combined)
        return output
