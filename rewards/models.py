import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepNet(nn.Module):
    """
    Base Class for all the models that will be added from here and inherited from this class.
    """

    def __init__(self) -> None:
        super(DeepNet, self).__init__()

    def save(
        self, filename: str = "model.pth", folder_path: Optional[str] = None
    ) -> None:
        """
        Save the model to a file.

        Args:
            filename (str, optional): The file name. Defaults to "model.pth".
            folder_path (str, optional): The folder path to save the model. Defaults to "None".
        Returns:
            None
        """
        if folder_path is None:
            folder_path = "./models"
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)

            filename = os.path.join(folder_path, filename)
            torch.save(self.state_dict(), filename)
        else:
            filename = os.path.join(folder_path, filename)
            torch.save(self.state_dict(), filename)
        print(f"=> model saved as: {filename}")

    def load(self, filename: str, folder_path: Optional[str] = None) -> None:
        """
        Load the model from a file.

        Args:
            filename (str): The file name.
            folder_path (str, optional): The folder path to save the model. Defaults to "None".
        Returns:
            None
        """
        try:
            model_path = (
                filename
                if folder_path is None
                else os.path.join(folder_path, filename)
            )
            self.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.eval()
        except Exception as e:
            print(e)
            print(f"=> model not found at: {model_path}")


class LinearQNet(DeepNet):
    def __init__(self, layers_conf: List[List[int]]):
        """Basic LinearQNet for the agent model.

        Args:
            layers_conf (List[List[int]]): The list of layers. Each element will be a list
            example: [[in_dim, out_dim], [in_dim, out_dim]]
        """
        super(LinearQNet, self).__init__()
        self.layers_conf = layers_conf
        self.num_layers = len(layers_conf)
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(layers_conf[i][0], layers_conf[i][1]))

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = F.relu(self.layers[i](x))
        return self.layers[-1](x)
