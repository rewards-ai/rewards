from collections import deque
import numpy as np
import random

class Memory:
    def __init__(
        self, 
        max_memory: int = int(1e5), 
        batch_size: int = 100
    ) -> None:
        """Intialized Memory class which is used to store data in a deque for training the models

        Args:
            max_memory (int, optional): The maximum number of elements the data can store. Defaults to int(1e5).
            batch_size (int, optional): size of the batch used for training. Defaults to 100.
        """
        self.max_memory = max_memory
        self.batch_size = batch_size
        
        self.data = deque(maxlen=self.max_memory)
        
    def get_batch(self) -> list:
        """Returns random sample of data from memory of size "self.batch_size"

        Returns:
            batch (list): batch of size "batch_size" from memory
        """
        if len(self.data) > self.batch_size:
            sample = random.sample(self.data, self.batch_size)
        else:
            sample = self.data
        return sample
        
    def store_memory(
            self, 
            state: np.ndarray, 
            action: list, 
            rewards: int, 
            next_state: np.ndarray, 
            done: bool
        ) -> None:
        """Stores a memory tuple containing the state, action, reward, next state, and done flag in the data.

        Args:
            state (np.ndarray): state of the environment.
            action (list): The action taken in the state.
            rewards (int): The rewards received for the actions by reward_function.
            next_state (np.ndarray): The next state after taking the action.
            done (bool): The done flag indicating if the episode terminated after the action.
        """
        self.data.append((state, action, rewards, next_state, done))
    
    def clear_memory(self) -> None:
        """Clears the memory
        """
        self.data.clear()