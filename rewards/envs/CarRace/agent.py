import os
import torch
import random
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional, Union

# TODO: Use DinoConf for configuration management afterwards

from .. import QTrainer

@dataclass
class DefaultConf:
    MAX_MEMORY: int = 100000
    BATCH_SIZE: int = 1000
    MODEL_NAME: str = "default_model.pth"
    PARENT_PATH: str = Path(__file__).parent.parent.parent
    DEVICE: str = "cpu"


class CarAgent:
    def __init__(
        self,
        model: torch.nn.Module,
        load_last_checkpoint: Optional[bool] = False,
        lr: float = 0.01,
        epsilon: float = 0.25,
        gamma: float = 0.9,
    ):
        """The Agent class which acts as a RL agent similar like Open AI's gym agent

        Args:
            model (torch.nn.Module): _description_
            load_last_checkpoint (Optional[bool], optional): _description_. Defaults to False.
            lr (float, optional): _description_. Defaults to 0.01
            epsilon (float, optional): _description_. Defaults to 0.25.
            gamma (float, optional): _description_. Defaults to 0.9.
        """
        self.n_games = 0
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma

        self.default_conf = DefaultConf()

        self.memory = deque(maxlen=self.default_conf.MAX_MEMORY)
        self.model = model
        if load_last_checkpoint:
            self.model.load_state_dict(
                torch.load(
                    os.path.join(self.default_conf.PARENT_PATH, "saved_models/default_model.pth"),
                    map_location=self.default_conf.DEVICE,
                )
            )
            self.model.eval()

        self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma)
    
    def get_state(self, game) -> np.ndarray:
        """Returns the current state of the game
        rewards.ai currently supports two games [Snake, Car Race]. Hence each of the game state will be a numpy array representing the game current playground

        Args:
            game (_type_): _description_

        Returns:
            np.ndarray: _description_
        """
        state = game.radars
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        """Trains the agent for a longer step saving state-actions to the memory

        Returns:
            None
        """
        if len(self.memory) > self.default_conf.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.default_conf.BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Trains the agent for a single step without any memory search

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
            done (function): _description_

        Returns:
            _type_: _description_
        """
        return self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.epsilon = 25
        final_move = [0, 0, 0]
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def train_step(self, game):
        """
        Defines a single train step for an agent where the agent performs 
        some action in a given state to get next state, current rewards, and 
        its status of completion
        
        Args:
            game (_type_): _description_

        Returns:
            (_type_): _description_

        """
        state_old = self.get_state(game)
        final_move = self.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = self.get_state(game)
        self.train_short_memory(state_old, final_move, reward, state_new, done)
        self.remember(state_old, final_move, reward, state_new, done)

        return reward, done, score


if __name__ == "__main__":
    conf = DefaultConf()
    print(conf)
