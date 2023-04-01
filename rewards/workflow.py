import inspect
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import pygame
import torch
import wandb

from .agent import Agent
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
    EXPERIMENT_NAME: str = "sample RL experiment"

    # Environment configuration

    ENVIRONMENT_NAME: str = "car-race"
    ENVIRONMENT_WORLD: Union[str, int] = 1

    # Game configuration
    MODE: str = "training"
    CONTROL_SPEED: float = 0.05
    TRAIN_SPEED: int = 100
    SCREEN_SIZE: Optional[Tuple] = (800, 700)

    # Training configuration
    LR: float = 0.01
    LOSS: str = "mse"
    OPTIMIZER: str = "adam"

    # RL Configuration
    GAMMA: float = 0.99
    EPSILON: float = 0.99

    # Model configuration
    LAYER_CONFIG: Union[List[List[int]], torch.nn.Module] = None # required 

    CHECKPOINT_PATH: Optional[str] = None
    REWARD_FUNCTION: Callable = None # required 


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
            checkpoint_path=self.config.CHECKPOINT_PATH,
        )

        # Once everything is done then upload all configurations to wandb

        wandb_config = self.config.__dict__.copy()
        wandb_config["REWARD_FUNCTION"] = inspect.getsource(
            self.config.REWARD_FUNCTION if self.config.REWARD_FUNCTION is not None else default_reward_function
        )

        if isinstance(self.model, torch.nn.Module):
            wandb_config.pop("LAYER_CONFIG")
        wandb_config.pop("CHECKPOINT_PATH")

        self.run = wandb.init(
            project=self.config.EXPERIMENT_NAME, config=wandb_config
        )
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

        # Build Game
        # TODO:
        # For now we are assuming that we only have just one game and so we are keeping
        # all the game and env config at one place. In next set of version this will be
        # different as we will support it for multiple pre-built envs and custom envs

        # Build PyGame
        self.screen = pygame.display.set_mode(
            self.config.SCREEN_SIZE, pygame.HIDDEN
        )
        
        reward_func = self.config.REWARD_FUNCTION if self.config.REWARD_FUNCTION is not None else default_reward_function
        self.game = CarGame(
            frame=self.screen,
            track_num=self.config.ENVIRONMENT_WORLD,
            reward_func=reward_func,
        )

    def run_single_episode(self):
        total_score, record = 0, 0

        try:
            while True:
                time.sleep(0.01)
                pygame.display.update()

                self.game.FPS = self.config.TRAIN_SPEED
                reward, done, score = self.agent.train_step(self.game)
                self.game.timeTicking()

                if done:
                    self.game.initialize()
                    self.agent.n_games += 1
                    self.agent.train_long_memory()

                    if score > record:
                        self.agent.model.save()
                        record = score

                    total_score += score

                    if self.agent.n_games != 0:
                        self.run.log(
                            {
                                "episode score": score,
                                "mean score": total_score / self.agent.n_games,
                            }
                        )

                self.run.log({"Num Games": self.agent.n_games})

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        break
        except pygame.error:
            print("pygame error")
