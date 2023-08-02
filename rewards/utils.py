import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    """Custom RMSE or Root Mean Sqaured Error loss class
    """
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, input, target):
        return torch.sqrt(torch.mean((input - target)**2))

class MAELoss(nn.Module):
    """Custom MAE or Mean Absolute Error loss class
    """
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, input, target):
        return torch.mean(torch.abs(input - target))
    
class MSELoss(nn.Module):
    """Custom MSE or Mean Sqaured Error loss class
    """
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, input, target):
        return torch.mean((input - target)**2)