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

    def __init__(self) -> None:
        super(DeepNet, self).__init__()

    def save(
        self, checkpoint_folder_path : Optional[str] = None, model_name : Optional[str] = None, device : Optional[str] = "cpu"
    ) -> None:
        """Loads the model in a robust file structure and setting 

        Args:
            checkpoint_folder_path (Optional[str], optional): The model checkpoint folder where all the checkpoints are been saved. Defaults to None.
            model_name (Optional[str], optional): The name of the model to save. Defaults to None.
        """
        folder_path = DEFAULT_MODEL_PATH if checkpoint_folder_path is None else checkpoint_folder_path
        _model_name = f'model_{datetime.datetime.now()}_.pth' if model_name is None else model_name
        
        if folder_path == DEFAULT_MODEL_PATH and not os.path.exists(DEFAULT_MODEL_PATH):
            os.mkdir(DEFAULT_MODEL_PATH)
            print("=> Creating DEFAULT MODEL FOLDER PATH to saave all the model checkpoints")
        
        torch.save(
            self.state_dict(), 
            os.path.join(folder_path, _model_name)
        )
        print(f"=> Latest model saved as {_model_name}")
    
    def _get_file_list_mod_by_date(self, search_dir : str, reversed : Optional[bool] = True) -> List[str]:
        """Lists the files based on the date modified 

        Args:
            search_dir (str): The folder to search and sort 
            reversed (Optional[bool], optional): Wants in descending order. Defaults to True.

        Returns:
            List[str]: The list of sorted files 
        """
        files = list(filter(os.path.isfile, glob.glob(search_dir + "*")))
        files.sort(key=lambda x: os.path.getmtime(x), reverse=reversed)
        return files 
    
    def load(self, checkpoint_folder_path : Optional[str] = None, model_name : Optional[str] = None, device : Optional[str] = "cpu") -> None:
        """Loads the model in a robust file structure and setting 

        Args:
            checkpoint_folder_path (Optional[str], optional): The model checkpoint folder where all the checkpoints are been saved. Defaults to None.
            model_name (Optional[str], optional): The name of the model to save. Defaults to None.
        """
        if checkpoint_folder_path and len(os.listdir(checkpoint_folder_path)) > 0:
            print(checkpoint_folder_path, self._get_file_list_mod_by_date(checkpoint_folder_path))
            _model_name = self._get_file_list_mod_by_date(checkpoint_folder_path)[0] if model_name is None else model_name 
            model_path = os.path.join(checkpoint_folder_path, _model_name)
            try:
                self.load_state_dict(
                    torch.load(
                        model_path,
                        map_location=device,
                    )
                )
                self.eval()
                print(f"=> Model loaded successfully from {model_path}")
                
            except Exception as e:
                # TODO: Write custom error message from exceptions and logging 
                print(f"=> Error occured {e}")
                
        # give the options to load from the default checkpoint folder path and files 
        # for this version the default checkpoint folder path will be ./saved_models and the latest model will be taken in consideration 
        
        elif os.path.exists(DEFAULT_MODEL_PATH):
            if len(os.listdir(DEFAULT_MODEL_PATH)) > 0: 
                _default_model_path = os.path.join(
                    DEFAULT_MODEL_PATH, 
                    self._get_file_list_mod_by_date(DEFAULT_MODEL_PATH)[0]
                )
                try:
                    self.load_state_dict(
                        torch.load(_default_model_path, map_location=device)
                    )
                    self.eval() 
                    print(f"=> Found default location and loaded successfully from {_default_model_path}")
                except Exception as e:
                    print(f"=> No model found  Exception occured at: {e}")
        else:
            print("=> No location or default location found")
        

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
