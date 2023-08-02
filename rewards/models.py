# coding=utf-8
# Copyright 2023-present rewards.ai. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import datetime 
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



DEFAULT_MODEL_PATH = './saved_models/'

class DeepNet(nn.Module):
    """
    Base Class for all the models that will be added from here and inherited from this class.
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initializing DeepNet

        Args:
            verbose (bool, optional): verbose if true, display current scores else does not. Defaults to False.
        """
        self.verbose = verbose
        super(DeepNet, self).__init__()

    def save(self, model_name: str, checkpoint_folder_path : Optional[str] = None) -> None:
        """Loads the model in a robust file structure and setting 

        Args:
            model_name (str): The name of the model to save.
            checkpoint_folder_path (str, optional): The model checkpoint folder where all the checkpoints are been saved. Defaults to None.
        """
        folder_path = DEFAULT_MODEL_PATH if checkpoint_folder_path is None else checkpoint_folder_path
        model_name = f"model-{model_name}.pth"
        model_path = os.path.join(folder_path, model_name)
        
        try:
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
                if self.verbose: print(f"=> Creating {folder_path} folder to save all the model checkpoints.")
            
            torch.save(self.state_dict(), model_path)
            if self.verbose: print(f"=> Model saved to {model_path}")
        except Exception as e:
            print(f"Some error occored while saving model:\n{e}")
    
    def load(self, model_name: str, checkpoint_folder_path: Optional[str] = None, device : Optional[str] = "cpu") -> None:
        """Loads the model in a robust file structure and setting 

        Args:
            model_name (str): The name of the model to save.
            checkpoint_folder_path (str, optional): The model checkpoint folder where all the checkpoints are been saved. Defaults to None.
            device (str, optional): mentions the device used to load model (cpu or gpu). Default is "cpu".
        """
        
        folder_path = DEFAULT_MODEL_PATH if checkpoint_folder_path is None else checkpoint_folder_path
        model_name = f"model-{model_name}.pth"
        model_path = os.path.join(folder_path, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} file does not exist.")
        
        try:            
            self.load_state_dict(torch.load(model_path, map_location=device))
            self.eval()
            if self.verbose: print(f"=> Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Some error occored while loading model:\n{e}")

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
