import sys 
import time 
import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd
import pygame
import torch
import wandb

from .envs.car import CarGame
from .models import LinearQNet
from .trainer import QTrainer

# TODO: Make a video recording feature (that will record and upload to W&B dashboard once training is complete)
# TODO: Things that is to be tracked in wandb

# - Live metrics of the plots
# - CPU usages (default)
# - All the configurations
# - Once experiment is complete then upload the recorded pygame environment


def default_reward_function(props):
    if props["isAlive"]:
        return 1
    return 0


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

class RLWorkFlow:
    def __init__(
        self, experiment_configuration: Optional[WorkFlowConfigurations] = None
    ) -> None:
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
            self.model = LinearQNet(self.config.LAYER_CONFIG) if self.config.LAYER_CONFIG is not None else LinearQNet([[5, 64], [64, 3]])

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

        # Once everything is done then upload all configurations to wandb
        
        if self.config.ENABLE_WANDB:
            wandb_config = self.config.__dict__.copy()
            wandb_config["REWARD_FUNCTION"] = inspect.getsource(
                self.config.REWARD_FUNCTION if self.config.REWARD_FUNCTION is not None else default_reward_function
            )

            if isinstance(self.model, torch.nn.Module):
                wandb_config.pop("LAYER_CONFIG")
                # Also upload the model to wandb artifact 
                
            wandb_config.pop("CHECKPOINT_FOLDER_PATH")

            self.run = wandb.init(
                project=self.config.EXPERIMENT_NAME, config=wandb_config
            )
            
        if self.config.ENABLE_WANDB:
            config_dataframe = pd.DataFrame(
                data={
                    "configuration name": list(wandb_config.keys()),
                    "configuration": [
                        str(ele) for ele in list(wandb_config.values())
                    ],
                }
            )
        
            config_table = wandb.Table(dataframe=config_dataframe)
            config_table_artifact = wandb.Artifact(
                "configuration_artifact", type="dataset"
            )
            config_table_artifact.add(config_table, "configuration_table")

            self.run.log({"Configuration": config_table})
            self.run.log_artifact(config_table_artifact)

        
        self.reward_func = self.config.REWARD_FUNCTION if self.config.REWARD_FUNCTION is not None else default_reward_function

    def stop_game(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
    
    def run_episodes(self):
        # Build PyGame
        self.screen = pygame.display.set_mode(
            self.config.SCREEN_SIZE, pygame.HIDDEN
        )
        
        self.game = CarGame(
            frame=self.screen,
            track_num=self.config.ENVIRONMENT_WORLD,
            reward_func=self.reward_func,
        )
        
        
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
                
                if self.agent.n_games != 0 and self.config.ENABLE_WANDB:
                    self.run.log(episode_response)
                yield episode_response
            
            except pygame.error:
                pygame.quit() 
                break 
        print("=> Process finished")