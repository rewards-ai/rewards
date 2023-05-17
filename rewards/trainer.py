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
import random
from typing import Any, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from .agent import Agent

# TODO:
# - Move RMSE, MAE to utils module


class RootMeanSquaredError(torch.nn.Module):
    def __init__(self):
        """Root mean squared error function in PyTorch"""
        super(RootMeanSquaredError, self).__init__()

    def forward(self, x, y):
        return torch.sqrt(torch.mean((x - y) ** 2))


class MeanAbsoluteError(torch.nn.Module):
    def __init__(self):
        """
        Mean absolute error function in PyTorch
        """
        super(MeanAbsoluteError, self).__init__()

    def forward(self, x, y):
        return torch.mean(torch.abs(x - y))


class QTrainer(Agent):
    def __init__(self, model, model_name, checkpoint_folder_path, lr=0.001, gamma=0.9, epsilon=0.2, optimizer="adam", loss="mse"):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.model_name = model_name

        self.model = deepcopy(model)
        loss_fn, optimizer_info = self._get_loss_optimizer_info(
            loss, optimizer
        )
        self.criterion = loss_fn()
        self.optimizer = optimizer_info(self.model.parameters(), lr=self.lr)
        self.checkpoint_folder_path = checkpoint_folder_path

        super(QTrainer, self).__init__(
            model=self.model,
            lr=self.lr,
            epsilon=self.epsilon,
            gamma=self.gamma,
            checkpoint_folder_path=self.checkpoint_folder_path, 
            model_name=self.model_name
        )

    def _get_loss_optimizer_info(
        self, loss: str, optimizer: str
    ) -> List[Union[int, str, float]]:
        """_summary_

        Args:
            loss (str): _description_
            optimizer (str): _description_

        Returns:
            List[str, Union[int, str, float]]: _description_
        """
        loss_info = {
            "mse": torch.nn.MSELoss,
            "rmse": RootMeanSquaredError,
            "mae": MeanAbsoluteError,
        }

        optimizer_info = {
            "adam": optim.Adam,
            "rmsprop": optim.RMSprop,
            "adagrad": optim.Adagrad,
        }

        return loss_info[loss], optimizer_info[optimizer]

    def step(
        self,
        state: Any,
        action: Union[np.ndarray, List[Union[float, int]]],
        reward: Union[float, int],
        next_state: Any,
        done: bool,
    ) -> None:
        """Single step function for a single episode for the agent's training

        Args:
            state (Any): The current state of the environment
            action (Union[np.ndarray, List[Union[float, int]]]): The action taken by the agent
            reward (Union[float, int]): The reward that the agent gets
            next_state (Any): Next state after the action taken by the agent
            done (bool): Whether the game terminates or not
        """
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, dim=0)
            next_state = torch.unsqueeze(next_state, dim=0)
            action = torch.unsqueeze(action, dim=0)
            reward = torch.unsqueeze(reward, dim=0)
            done = (done,)

        state_prediction = self.model(state)
        state_target = (
            state_prediction.clone()
        )

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(
                    self.model(next_state[idx])
                )
            state_target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(state_target, state_prediction)
        loss.backward()
        self.optimizer.step()

    def train_long_memory(self) -> None:
        """Trains the agent for a longer step saving state-actions to the memory
        
        Returns:
            None
        """
        if len(self.memory) > self.BATCH_SIZE:
            mini_sample = random.sample(self.memory, self.BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.step(states, actions, rewards, next_states, dones)

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
        return self.step(state, action, reward, next_state, done)

    def train_step(self, i, game) -> List[Union[int, float, bool]]:
        """
        Defines a single train step for an agent where the agent performs
        some action in a given state to get next state, current rewards, and
        its status of completion

        Args:
            game (Any): The game environment

        Returns:
            (List[Union[int, float], bool, Union[int, float]]) : [current_reward, done, score]
        """

        state_old = self.get_state(i, game)
        final_move = self.get_action(state_old)
        
        reward, done, pixel_data = game.step(i, final_move)
        
        state_new = self.get_state(i, game)
        self.train_short_memory(state_old, final_move, reward, state_new, done)
        self.remember(state_old, final_move, reward, state_new, done)

        return reward, done, pixel_data
