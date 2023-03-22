import os 
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Union, Tuple, List, Dict 

import matplotlib.pyplot as plt
import pygame

plt.ion()
pygame.init()

def convert_points(point):
    return int(point[0]), int(point[1])

def get_test_params():
    return {
        "isAlive": None,
        "obs": [],
        "dir": None,
        "rotationVel": None
    }

# Configurations for both the game and the car 

@dataclass
class GameConfig:
    
    # Global Path
    PARENT_PATH: str = Path(__file__).parent.parent.parent
    
    # Initial Car and Game Configuration
    ASSET_PATH : str = str(os.path.join(PARENT_PATH, 'assets/CarRace'))
    CAR_SCALE : int = 500 
    SCREEN_SIZE : Tuple[int, int] = field(default_factory = lambda : (800, 700))
    CAR_TRACKS : Dict[int, str] = field(default_factory=lambda : {
        0 : "track_test_7.png", 
        1 : "track_test_4.png", 
        2 : "track_test_5.png"  
    })
    CAR_IMAGE : str = "car.png"
    
    # Car Configuration
    DRIVE_FACTOR : int = 12 
    CAR_FPS : int = 15 
    CAR_ANGLE : int = 0 
    
    CAR_RECT_SIZE : Tuple[int, int] = field(default_factory = lambda : (200, 100))
    CAR_VELOCITY_VECTOR : Tuple[float, float] = field(default_factory = lambda : (0.8, 0.0))
    CAR_ROTATION_VELOCITY : Union[int, float] = 15
    CAR_DIRECTION : Union[int, float] = 0
    CAR_IS_ALIVE : bool = True 
    CAR_REWARD : int = 0, 
    CAR_RADAR : List[Union[int, float]] = field(default_factory = lambda : [0, 0, 0, 0, 0])
    

class Track:
    def __init__(self, track_num : int = 0):
        """Car Track which is a PyGame under which the agent will operate 

        Once intialized all the selected configs will return a json and will save inside the local
        Args:
            track_num (int, optional): Which track to choose. Track Options [0, 1, 2]. Defaults to 0.
        """
        self.game_conf = GameConfig() 
        self.track_image_path = os.path.join(
            self.game_conf.ASSET_PATH, self.game_conf.CAR_TRACKS[track_num]
        )
        
        self.car_image_path = os.path.join(
            self.game_conf.ASSET_PATH, self.game_conf.CAR_IMAGE
        )
        
        self.track_image = pygame.image.load(self.track_image_path)
        self.car_image = pygame.transform.scale(
            pygame.image.load(self.car_image_path), (self.game_conf.CAR_SCALE, self.game_conf.CAR_SCALE)
        )
        
    def track(self):
        return self.track_image

    def track_checkpoint(self, car):
        visited = []


class CarGame(Track):
    def __init__(self, reward_func : Optional[Union[str, Any]] = None, frame : Optional[Any] = None, track_num : Optional[int] = 0):
        super(CarGame, self).__init__(track_num=track_num) 
        
        self.screen = pygame.display.set_mode(self.game_conf.SCREEN_SIZE)
        self.screen.fill((0, 0, 0))
        self.params = {
            "isAlive": None,
            "obs": None,
            "dir": None,
            "rotationVel": None
        }
        
        # TODO: Change this (car_image, original_image, image confusion and make proper usage and naming of the variables)
        
        self.angle = self.game_conf.CAR_ANGLE
        self.original_image = self.car_image
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.game_conf.CAR_RECT_SIZE)
        self.vel_vector = pygame.math.Vector2(self.game_conf.CAR_VELOCITY_VECTOR) 
        self.rotation_vel = self.game_conf.CAR_ROTATION_VELOCITY
        
        self.direction = self.game_conf.CAR_DIRECTION
        self.drive_factor = self.game_conf.DRIVE_FACTOR
        self.alive = self.game_conf.CAR_IS_ALIVE
        self.reward = 0
        self.radars = self.game_conf.CAR_RADAR
        
        self.clock = pygame.time.Clock() 
        self.reward_func = reward_func if reward_func is not None else self._default_reward_func 
        self.track = self.track_image
        self.FPS = self.game_conf.CAR_FPS 
        self.iterations = 0 
        
    
    def initialize(self):
        """Initializes the game state from where it had started
        """
        self.screen = pygame.display.set_mode(self.game_conf.SCREEN_SIZE)
        
        self.angle = self.game_conf.CAR_ANGLE
        self.original_image = self.car_image
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.game_conf.CAR_RECT_SIZE)
        self.vel_vector = pygame.math.Vector2(self.game_conf.CAR_VELOCITY_VECTOR) 
        self.rotation_vel = self.game_conf.CAR_ROTATION_VELOCITY
        
        self.direction = self.game_conf.CAR_DIRECTION
        self.drive_factor = self.game_conf.DRIVE_FACTOR
        self.alive = self.game_conf.CAR_IS_ALIVE
        self.reward = 0
        self.radars = self.game_conf.CAR_RADAR
        
    
    def _default_reward_func(self, props):
        """Default reward function initialised if the user does not provide any reward function 

        Args:
            props (_type_): properties as arguments containing a dict of parameter 

        Returns:
            _type_: _description_
        """
        if not props['isAlive']:
            return 1 
        else:
            return 0 
    
    def draw(self):
        self.screen.blit(self.track, (0, 0))
        self.screen.blit(self.image, self.rect.topleft)
    
    def timeTicking(self):
        self.clock.tick(self.FPS)
    
    def play_step(self, action):
        self.iterations += 1 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        self.draw() 
        self.radars = [0, 0, 0, 0, 0]
        self.drive()
        
        # Change from if-else to ActionSwitch().action_response(action)

        if action[0] == 1:
            self.direction = -1
        elif action[1] == 1:
            self.direction = 1
        elif action[2] == 1:
            self.direction = 0
            self.drive()
        elif action[3] == 1:
            self.vel_vector.scale_to_length(0.8)
            self.rotation_vel = 15
        elif action[4] == 1:
            self.vel_vector.scale_to_length(1.2)
            self.rotation_vel = 10
        elif action[5] == 1:
            self.vel_vector.scale_to_length(1.6)
            self.rotation_vel = 7
        else:
            self.direction = 0
        
        self.rotate() # to be implemented
        self.collision() # to be implemented

        for i, radar_angle in enumerate((-60, -30, 0, 30, 60)):
            self.radar(i, radar_angle) # to be implemented 

        if self.radars[0] < 15 and self.radars[4] < 15 and self.radars[1] < 25 and self.radars[2] < 25 and self.radars[3] < 25:
            self.alive = False
        else:
            self.alive = True

        self.params = {
            "isAlive": self.alive,
            "obs": self.radars,
            "dir": self.direction,
            "rotationVel": self.rotation_vel
        }

        reward = self.reward_func(self.params)

        self.reward += reward
        return reward, not self.alive, self.reward
        
        
    def drive(self):
        self.rect.center += self.vel_vector * 12
    
    def collision(self):
        """_summary_
        """
        length = 20
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        try:
            if self.screen.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) \
                    or self.screen.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
                self.alive = False

            pygame.draw.circle(self.screen, (0, 255, 255, 0), collision_point_right, 4)
            pygame.draw.circle(self.screen, (0, 255, 255, 0), collision_point_left, 4)
        except:
            self.alive = False 
    
    def rotate(self):
        """_summary_
        """
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)
    
    def radar(self, i, radar_angle):
        """_summary_

        Args:
            i (_type_): _description_
            radar_angle (_type_): _description_
        """
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])
        try:
            while not self.screen.get_at((x, y)) == pygame.Color(2, 105, 31, 255) and length < 200:
                length += 1
                x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
                y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

            pygame.draw.line(self.screen, (255, 255, 255, 255), self.rect.center, (x, y), 1)
            pygame.draw.circle(self.screen, (0, 255, 0, 0), (x, y), 3)

            dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2) + math.pow(self.rect.center[1] - y, 2)))
            self.radars[i] = dist
        except:
            self.alive = False
    
    
    def play_human(self):
        action = [0, 0, 0, 0, 0, 0]
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            action[0] = 1
        elif keys[pygame.K_d]:
            action[1] = 1
        elif keys[pygame.K_w]:
            action[2] = 1
        elif keys[pygame.K_1]:
            action[3] = 1
        elif keys[pygame.K_2]:
            action[4] = 1
        elif keys[pygame.K_3]:
            action[5] = 1
        self.play_Step(action)

    
    def train(self, mode, control_speed, train_speed, agent):
        """_summary_

        Args:
            mode (_type_): _description_
            control_speed (_type_): _description_
            train_speed (_type_): _description_
            agent (_type_): _description_

        Returns:
            _type_: _description_
        """
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0 
        
        while True:
            pygame.display.update()
            x = pygame.surfarray.array3d(self.screen)
            
            if mode == "human":
                time.sleep(control_speed)
                self.play_human()
            
            else:
                self.FPS = train_speed
                reward, done, score = agent.train_step(self)
                self.timeTicking() 

                if done: 
                    self.initialize() 
                    agent.n_games += 1
                    agent.train_long_memory()
                    
                    if score > record:
                        record = score 
                        agent.model.save() 
                    print('Game', agent.n_games, 'Score', score, 'Record:', record)
                    plot_scores.append(score)
                    total_score += score
                    mean_score = total_score / agent.n_games
                    plot_mean_scores.append(mean_score)
                    
                    plt.clf()
                    plt.title('Training...')
                    plt.xlabel('Number of Games')
                    plt.ylabel('Score')
                    plt.plot(plot_scores)
                    plt.plot(plot_mean_scores)
                    plt.ylim(ymin=0)
                    plt.text(len(plot_scores) - 1, plot_scores[-1], str(plot_scores[-1]))
                    plt.text(len(plot_mean_scores) - 1, plot_mean_scores[-1], str(plot_mean_scores[-1]))
                    plt.show(block=False)
                    plt.pause(.1)
                
    
        
class ActionSwitch(CarGame):
    # TODO: 
    # - Propper testing is needed 
    # - Configuration management is needed for the changes in vel_vector, rotation_vector, etc
    
    def __init__(self) -> None:
        super(ActionSwitch, self).__init__() 
    
    def action_response(self, action):
        self.direction = 0
        return getattr(self, 'case_' + str(action), lambda : self.direction)() 

    def case_0(self):
        self.direction = -1 
    
    def case_1(self):
        self.direction = 1
    
    def case_2(self):
        self.direction = 0
        self.drive() 
    
    def case_3(self):
        self.vel_vector.scale_to_length(0.8)
        self.rotation_vel = 15 
    
    def case_4(self):
        self.vel_vector.scale_to_length(1.2)
        self.rotation_vel = 10
    
    def case_5(self):
        self.vel_vector.scale_to_length(1.6)
        self.rotation_vel = 7