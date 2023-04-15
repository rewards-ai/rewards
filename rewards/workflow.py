import cv2 
import sys 
import time 
import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import pandas as pdsys
import pygame
import torch

from .envs.car import CarGame
from .models import LinearQNet
from .trainer import QTrainer


@dataclass(kw_only=True)
class WorkFlowConfigurations:
    # wandb experiment
    DEVICE : str = "cpu"
    # Environment configuration

    ENVIRONMENT_NAME: str = "car-race"
    ENVIRONMENT_WORLD: Union[str, int] = 1

    # Game configuration
    MODE: str = "training"
    CAR_SPEED: int = 100
    SCREEN_SIZE: Optional[Tuple] = (800, 700)

    # Training configuration
    LR: float = 0.01
    LOSS: str = "mse"
    OPTIMIZER: str = "adam"
    NUM_EPISODES : int = 100 

    # RL Configuration
    GAMMA: float = 0.99
    EPSILON: float = 0.99

    # Model configuration
    LAYER_CONFIG: Union[List[List[int]], torch.nn.Module] = None # required 

    CHECKPOINT_FOLDER_PATH: Optional[str] = None
    CHECKPOINT_MODEL_NAME: Optional[str] = None
    REWARD_FUNCTION: Callable = None # required 

    # Tracking configuration 
    ENABLE_WANDB : bool = False 
    
    # Newly added 
    
    MODE : str = "training"
    PYGAME_WINDOW_TYPE : str = "display"

class RLWorkFlow:
    def __init__(
        self, experiment_configuration: Optional[WorkFlowConfigurations] = None) -> None:
        """
        **RLWorkFlow** is the module which ables us to run complete RL experiments
        """

        self.config = (
            WorkFlowConfigurations()
            if experiment_configuration is None
            else experiment_configuration
        )

        # Build model
        if isinstance(self.config.LAYER_CONFIG, torch.nn.Module):
            self.model = self.config.LAYER_CONFIG
        else:
            self.model = LinearQNet(
                self.config.LAYER_CONFIG) if self.config.LAYER_CONFIG is not None else LinearQNet([[5, 64], [64, 3]])

        # Build Agent
        self.agent = QTrainer(
            lr=self.config.LR,
            gamma=self.config.GAMMA,
            epsilon=self.config.EPSILON,
            model=self.model,
            loss=self.config.LOSS,
            optimizer=self.config.OPTIMIZER,
            checkpoint_folder_path=self.config.CHECKPOINT_FOLDER_PATH,
            model_name = self.config.CHECKPOINT_MODEL_NAME
        )

        self.reward_func = self.config.REWARD_FUNCTION if self.config.REWARD_FUNCTION is not None else None 
        print("=> All configs done and saved")

        self.game = CarGame(
            mode=self.config.MODE, 
            track_num=self.config.ENVIRONMENT_WORLD, 
            reward_function=self.reward_func, 
            display_type=self.config.PYGAME_WINDOW_TYPE, 
            screen_size=self.config.SCREEN_SIZE
        )
    
    
    def stream_single_episode(self):
        done = False 
        while not done and self.config.PYGAME_WINDOW_TYPE == "surface":
            _, done, score, pixel_data = self.agent.train_step(self.game)
            yield {
                'data' : pixel_data, 
                'score' : score,
            }
                
    def stream_multi_episodes(self):
        for episode in range(1, self.config.NUM_EPISODES + 1):
            self.game.initialize()
            self.game.FPS = self.config.CAR_SPEED
            total_score, record = 0, 0
            
            for streamed_responses in self.stream_single_episode():
                cv2.imshow('Frame', streamed_responses['data'])
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    sys.exit() 
            
            score = streamed_responses['score']
            if score > record:
                    self.agent.model.save(
                        self.config.CHECKPOINT_FOLDER_PATH, 
                        self.config.CHECKPOINT_MODEL_NAME, self.config.DEVICE
                    )
                    record = score 
            total_score += score 
        print("=> Process finished")
        cv2.destroyAllWindows()
        
    def run_episodes(self):
        for episode in range(1, self.config.NUM_EPISODES + 1):
            self.game.initialize() # change it to env.reset() 
            self.game.FPS = self.config.CAR_SPEED
            total_score, record = 0, 0
            done = False 
            
            try:
                while not done:
                    time.sleep(0.01) 
                    pygame.display.update() # try to comment it and see what happens 
                    
                    _, done, score = self.agent.train_step(self.game)
                    self.game.timeTicking() 
                self.agent.n_games += 1 
                self.agent.train_long_memory() 
                
                if score > record:
                    self.agent.model.save(self.config.CHECKPOINT_FOLDER_PATH, self.config.CHECKPOINT_MODEL_NAME, self.config.DEVICE)
                    record = score 
                total_score += score 
                
                episode_response = {
                    "episode_num" : episode, 
                    "episode score": score,
                    "mean score": total_score / self.agent.n_games
                }
            
            except pygame.error:
                pygame.quit() 
                break 
        print("=> Process finished")