import os
import glob 
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch

@dataclass
class AgentConf:
    MAX_MEMORY: int = 100000
    BATCH_SIZE: int = 1000
    PARENT_PATH: str = str(Path(__file__).parent.parent)
    DEVICE: str = "cpu"


class Agent(AgentConf):
    def __init__(
        self,
        model: torch.nn.Module,
        checkpoint_folder_path: Optional[str] = None,
        model_name : Optional[str] = None, 
        lr: float = 0.01,
        epsilon: float = 0.25,
        gamma: float = 0.9,
    ) -> None:
        super(Agent, self).__init__()
        """The Agent class which acts as a RL agent similar like Open AI's gym agent

        Args:
            model (torch.nn.Module): _description_
            checkpoint_path (Optional[str], optional): The model checkpoint to load its weight. Defaults to False.
            lr (float, optional): _description_. Defaults to 0.01
            epsilon (float, optional): _description_. Defaults to 0.25.
            gamma (float, optional): _description_. Defaults to 0.9.
        """
        self.n_games = 0
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma

        self.memory = deque(maxlen=self.MAX_MEMORY)
        self.model = model
        
        # Lates changes loading the model directly if exists 
        self.model.load(checkpoint_folder_path, model_name, self.DEVICE)
        
    def get_state(self, game: Any) -> np.ndarray:
        """Returns the current state of the game.
        NOTE: Some Assumptions:
        - We assume that the game environment is made using pygame
        - We also assume that the agent inside the game uses `radars` that keeps track of its all position and other parameters.

        Args:
            game (rewards.env.car.CarGame): The current game environment of pygame.

        Returns:
            np.ndarray: An array of the state of the game. Here it is the agent's radars.
        """

        # TODO: Check the type of game
        state = game.radars
        return np.array(state, dtype=int)

    def remember(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, List[int]],
        reward: Union[int, float],
        next_state: np.ndarray,
        done: bool,
    ) -> List[Union[float, int]]:
        """Remmembers the state of the game for the exploration phase of the agent

        Args:
            state (np.ndarray): The current state of the agent
            action (Union[np.ndarray, List[int]]): Action taken by the agent
            reward (Union[int, float]): Reward that the agent gets
            next_state (np.ndarray): The next state which the agent takes after taking the current action
            done (bool): Whether the game is finished or not.

        Returns:
            List[Union[float, int]]: The final move after exploration which is an action
        """
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        """
        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.epsilon = 25 # self.epsilon (WHY NOT)
        final_move = [
            0,
            0,
            0,
        ]  # TODO: need to make a general array of zeros which matches with the length of action

        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
