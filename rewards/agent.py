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

"""
The AgentConf and Agent class are the two classes two create RL/DRL agents and train them on the provided environment. 

TODOs:
----- 
- Random.randint -> random.uniform(0, 1) with a SEED 
- Take the float value of random rather than converting it into int 
- Implement epsilon decay 
"""

import torch 
import random 
import numpy as np 
from pathlib import Path 
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional, Union 


@dataclass
class AgentConf:
    """
    The AgentConf class ia mainly designed to create plug-n-play agents easily with different configurations. All the different types of 
    configurations are stored inside this agent configurations in order to initialize an agent before the training. 
    args:
        MAX_MEMORY (int) The maximum memory that the agent should hold for exploitation 
        BATCH_SIZE (int) The number of batches of instances should be trained at once for an agent. 
        DEVICE (str) Specifies in which device (pytorch backend) to train the model. Defaults to "cpu". 
        CHECKPOINT_FOLDER_PATH (str) Specifies whetre to store the model checkpoints. Defaults to: ./saved_models 
        MODEL_NAME (str) Specifies the name of the model to store. Defaults to: model.pth 
        LEARNING_RATE (float) Specifies the learning rate for the agent 
        EPSILON (float) Specifies the initial probability value of epsilon for exploration and exploitation tradeoff.
        GAMMA (float) Specifies discount factor and quantifies how much importance we give for future rewards
    """
    MAX_MEMORY: int = 100000  
    BATCH_SIZE: int = 1000 
    DEVICE: str = "cpu" 
    CHECKPOINT_FOLDER_PATH : str = "./saved_models"
    MODEL_NAME : str = "model.pth"
    LEARNING_RATE : float = 0.01 
    EPSILON : float = 0.99
    GAMMA : float = 0.25 
    

class Agent:
    def __init__(self, model : torch.nn.Module, agent_conf : Optional[AgentConf] = None) -> None:
        """The Agent class. 
        Agent class is mainly responsible for creating plug-n-play RL/DRL agents with just few lines of code. You are required to 
        specify the configuration in which the agent will work, and based on that you can create different agents and track them all 
        at once. 

        Args:
            model (torch.nn.Module): Model architecture and parameters 
            agent_conf (Optional[AgentConf], optional): The configurations generated from AgentConf class. Defaults to None. (All the default configurations)
        """
        
        
class MultiAgent: 
    def __init__(self, models : List[torch.nn.Module], agent_confs : List[AgentConf]) -> None:
        """_summary_

        Args:
            models (List[torch.nn.Module]): _description_
            agent_confs (List[AgentConf]): _description_
        """
        raise NotImplementedError 