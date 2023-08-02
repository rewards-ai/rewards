import numpy as np
import torch
from copy import deepcopy
from rewards.memory import Memory
import random
from rewards.utils import MAELoss, RMSELoss, MSELoss
from torch.optim import Adam, RMSprop, Adagrad
from typing import Optional, Any, List, Union
from rewards.models import DeepNet

class DQN:
    def __init__(self, 
            model: DeepNet,
            lr: Optional[float] = 0.001,
            gamma: Optional[float] = 0.9, 
            epsilon: Optional[float] = 0.2,
            epsilon_decay_rate: Optional[float] = 0,
            optimzer: Optional[str] = "adam", 
            loss: Optional[str] = "mse",
            memory: Optional[Memory] = Memory()
        ) -> None:
        """Initializes the DQN algorithm base on configurations provided

        Args:
            model (DeepNet): a Deep Neural Network
            lr (float, optional): learning rate. Defaults to 0.001.
            gamma (float, optional): a q-learning hyperparameter for calculating new q value. Defaults to 0.9.
            epsilon (float, optional): It controls the exploration vs exploitation factor. Defaults to 0.2.
            epsilon_decay_rate (float, optional): Decays the value of epsilon to reduce exploration Defaults to 0, which means no decay.
            optimzer (str, optional): Optimizer algorithm to be used for training. Defaults to "adam".
            criterion (str, optional): Loss function to be used for training. Defaults to "mse".
            memory (Memory, optional): Memory class used for training.
        """
                
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        
        self.model = model
        self.memory = deepcopy(memory)
     
        self.criterion, self.optimizer = self._get_criterion_optimizer(loss, optimzer)
        
    def _get_criterion_optimizer(self, loss: str, optimzer: str) -> None:
        """Returns criterion function and optimizer function to be used for training from "loss" and "optimizer" variables

        Raises:
            ValueError: Raised when "loss" or "optimizer" are incorrectly mentioned

        Returns:
            loss_func : returns loss function module for calculating loss
            optimizer_func : returns optimizer function module for training purposes
        """
        loss_map = {
            "mse": MSELoss(),
            "rmse": RMSELoss(),
            "mae": MAELoss()
        }
        
        optim_map = {
            "adam": Adam,
            "rmsprop": RMSprop,
            "adagrad": Adagrad
        }
        
        if loss in loss_map.keys():
            loss_func = loss_map[loss]
        else:
            raise ValueError(f"\n Currenly only {loss_map.keys()} errors are supported.")
        
        if optimzer in optim_map.keys():
            optimzer_func = optim_map[optimzer]
        else:
            raise ValueError(f"\n Currenly only {optim_map.keys()} optimizers are supported.")
        
        return loss_func, optimzer_func(self.model.parameters(), lr=self.lr)
        
    def _update_model(self, state: np.ndarray,  action: list,  reward: int, next_state: np.ndarray,  done: bool) -> None:
        """Single step function for a single episode for the agent's training

        Args:
            state (np.ndarray): state of the environment.
            action (list): The action taken in the state.
            reward (int): The rewards received for the actions by reward_function.
            next_state (np.ndarray): The next state after taking the action.
            done (bool): The done flag indicating if the episode terminated after the action.
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
        state_target = (state_prediction.clone())

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
        
    def get_action(self, state: np.ndarray, num_actions: int):
        """Returns actions based on epsilon which controls the trade-off between exploration and
           exploitation. Larger epsilon means more random steps. epsilon_decay_rate, reduces the 
           epsilon each step, to reduce exploration.

        Args:
            state (np.ndarray): state of the environment.
            num_actions (int): number of actions the agent can take

        Returns:
            final_action: Returns action to be taken by the agent
        """
        final_action = [0] * num_actions
        
        if random.random() < self.epsilon:
            move = random.randint(0, num_actions-1)
            final_action[move] = 1
        else:
            state = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state)
            move = torch.argmax(prediction).item()
            final_action[move] = 1
            
        return final_action
    
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

        state_old = game.get_state(i)
        final_move = self.get_action(state_old, game.num_actions)
        
        reward, done, pixel_data = game.step(i, final_move)
        
        state_new = game.get_state(i)
        self._update_model(state_old, final_move, reward, state_new, done)
        self.memory.store_memory(state_old, final_move, reward, state_new, done)

        return reward, done, pixel_data
    
    def train_from_memory(self) -> None:
        """Trains model from a batch from memory. This is called after an agent dies.
        """
        
        batch = self.memory.get_batch()
        states, actions, rewards, next_states, dones = zip(*batch)        
        self._update_model(states, actions, rewards, next_states, dones)
        
        self._decay_epsilon()
        
    
    def predict_step(self, i: int, game: Any) -> List[Union[int, float, bool]]:
        """Predicts action for the given state and steps the game using this action

        Args:
            i (int): index of the agent from agents list
            game (Any): The game environment

        Returns:
            List[Union[int, float, bool]]: Returns (reward, done, pixel_data) from step function of the environment
        """
        action = [0] * game.num_actions
        input = torch.tensor(game.agents[i].radars, dtype=torch.float)
        output = self.model(input)
        action[torch.argmax(output).item()] = 1
        
        return game.step(i, action)
        
    def _decay_epsilon(self) -> None:
        """Decays the epsilon value by epsilon_decay_rate to reduce exploration.
        """
        self.epsilon -= self.epsilon_decay_rate