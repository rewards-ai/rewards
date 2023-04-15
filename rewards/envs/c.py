import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pygame

plt.ion()
pygame.init()

# TODO:
# - Add Hydra for General Game Configuration management (This will help for writing less code and also adding custom game environments)
# - Make a general class of Track that will contain all the Track related information about PyGame
# - We do not need a secondary screen. right now we have `pygame.display` but we need to replace it by `pygame.surface` in later versions
# - (IMPORTANT) Inhering dataclass is not working.


class CarConfig:
    def __init__(self):
        # Global Path
        self.PARENT_PATH: str = Path(__file__).parent.parent

        # Initial Car and Game Configuration
        self.ASSET_PATH: str = str(
            os.path.join(self.PARENT_PATH, "assets/CarRace")
        )
        self.CAR_SCALE: int = 500
        self.SCREEN_SIZE: Tuple[int, int] = (800, 700)
        self.CAR_TRACKS: Dict[int, str] = {
            0: "track-1.png",
            1: "track-2.png",
            2: "track-3.png",
        }

        self.CAR_IMAGE: str = "car.png"

        # Car Configuration
        self.DRIVE_FACTOR: int = 12
        self.CAR_FPS: int = 15
        self.CAR_ANGLE: int = 0

        self.CAR_RECT_SIZE: Tuple[int, int] = (200, 50)
        self.CAR_VELOCITY_VECTOR: Tuple[float, float] = (0.8, 0.0)
        self.CAR_ROTATION_VELOCITY: Union[int, float] = 15
        self.CAR_DIRECTION: Union[int, float] = 0
        self.CAR_IS_ALIVE: bool = True
        self.CAR_REWARD: int = (0,)
        self.CAR_RADAR: List[Union[int, float]] = [0, 0, 0, 0, 0]
        
        # pygame screen type 
        # there are two options ("surface", display)
        self.PYGAME_SCREEN_TYPE : str = "display"


######################### Track Class #########################


class Track(CarConfig):
    def __init__(self, track_num: int = 0):
        """Car Track which is a PyGame under which the agent will operate

        Once intialized all the selected configs will return a json and will save inside the local
        Args:
            track_num (int, optional): Which track to choose. Track Options [0, 1, 2]. Defaults to 0.
        """
        super(Track, self).__init__()

        print(self.ASSET_PATH)
        print(self.CAR_ANGLE)
        print(self.CAR_VELOCITY_VECTOR)
        print(self.CAR_TRACKS)

        self.track_image_path = os.path.join(
            self.ASSET_PATH, "training", self.CAR_TRACKS[track_num]
        ).replace(os.sep, '/')

        self.car_image_path = os.path.join(self.ASSET_PATH, self.CAR_IMAGE)

        self.track_image = pygame.image.load(self.track_image_path)
        self.car_image = pygame.transform.scale(
            pygame.image.load(self.car_image_path),
            (self.CAR_SCALE, self.CAR_SCALE),
        )

    def track(self):
        return self.track_image

    def track_checkpoint(self, car):
        visited = []


####################### CarGame #########################


class CarGame(Track):
    def __init__(
        self,
        track_num: Optional[int] = 0,
        reward_func: Optional[Union[str, Callable]] = None,
    ) -> None:
        """### CarGame Environment:

        This is the reward's CarGame environment. There are mainly 4 main types of operations under which this environment operates.
        Those are as follows:

        - `is_alive` : This parameter determine whether the game is finished or not
        - `observation` : This parameter will return the current state of the car which are the radar's angular values and direction.
        - `direction` : This parameter will return the direction of the car.
        - `rotational_velocity` : This parameter will return the current rotational velocity of the car.

        The reward will be calculated under these above parameters. Our default reward function calculates reward upon playing each step.
        Although it does not make the agent learn anything. So we expect users to use their custom reward function to make the
        car agent to gain max reward and make it go brumm brummm !!

        Rewards has three different training environments to make the agent learn. It also has 2 Test environments for testing.
        In the future version we will let user to make/design their own car training environment and train their agents their.

        Args:
            track_num (Optional[int], optional): The . Defaults to 0.
            reward_func (Optional[Union[str, Callable]], optional): The reward function. Defaults to None.
        """

        # we need to also keep in track with the track's default asset parent folder path and it's other paths
        # we also need to design a way where user can make their own environment asset and use them by sucessfully integrating them to rewards

        super(CarGame, self).__init__(track_num=track_num)

        if self.PYGAME_SCREEN_TYPE == "display":
            self.screen = pygame.display.set_mode(self.SCREEN_SIZE)
        else:
            self.screen = pygame.Surface(self.SCREEN_SIZE) 
            
        self.screen.fill((0, 0, 0))
        self.params = {
            "is_alive": None,
            "observation": None,
            "direction": None,
            "rotational_velocity": None,
        }
        

        # TODO: Change this (car_image, original_image, image confusion and make proper usage and naming of the variables)

        self.angle = self.CAR_ANGLE
        self.original_image = self.car_image
        self.image = pygame.transform.rotozoom(
            self.original_image, self.angle, 0.1
        )
        self.rect = self.image.get_rect(center=self.CAR_RECT_SIZE)
        self.vel_vector = pygame.math.Vector2(self.CAR_VELOCITY_VECTOR)
        self.rotation_vel = self.CAR_ROTATION_VELOCITY

        self.direction = self.CAR_DIRECTION
        self.drive_factor = self.DRIVE_FACTOR
        self.alive = self.CAR_IS_ALIVE
        self.reward = 0
        self.radars = self.CAR_RADAR

        self.clock = pygame.time.Clock()
        self.track = self.track_image
        self.FPS = self.CAR_FPS
        self.iterations = 0
        self.reward_func = self._default_reward_func if reward_func is None else reward_func

    def initialize(self) -> None:
        """Initializes the game state from where it had started"""
        # The below needs to change as this will do those flikering stuffs
        # self.screen = pygame.display.set_mode(self.SCREEN_SIZE)

        self.angle = self.CAR_ANGLE
        self.original_image = self.car_image
        self.image = pygame.transform.rotozoom(
            self.original_image, self.angle, 0.1
        )
        self.rect = self.image.get_rect(center=self.CAR_RECT_SIZE)
        self.vel_vector = pygame.math.Vector2(self.CAR_VELOCITY_VECTOR)
        self.rotation_vel = self.CAR_ROTATION_VELOCITY

        self.direction = self.CAR_DIRECTION
        self.drive_factor = self.DRIVE_FACTOR
        self.alive = self.CAR_IS_ALIVE
        self.reward = 0
        self.radars = self.CAR_RADAR

    def _did_quit(self):
        """
        Quits the game when user presses the 'quit'/'close' key.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def _default_reward_func(self, props: Dict[str, Any]) -> int:
        """Default reward function initialised if the user does not provide any reward function
        NOTE: This function does not let the agent train or learn anything. It is just a dummy reward function

        Args:
            props (Dict[str, Any]): properties as arguments containing a dict of parameter

        Returns:
            int: The reward
        """
        # TODO: Change the default reward function, this is the actual reward function 
        
        reward = 0
        if props["isAlive"]:
            reward = 1
        obs = props["obs"]
        if obs[0] < obs[-1] and props["dir"] == -1:
            reward += 1
            if props["rotationVel"] == 7 or props["rotationVel"] == 10:
                reward += 1
        elif obs[0] > obs[-1] and props["dir"] == 1:
            reward += 1
            if props["rotationVel"] == 7 or props["rotationVel"] == 10:
                reward += 1
        else:
            reward += 0
            if props["rotationVel"] == 15:
                reward += 1
        return reward

    def _did_collide(self):
        """Checks the status whether the car collied or not
        If the car collides, then `isAlive` is False and game terminates.
        """

        # TODO: This function needs to be checked

        length = 20  # parameter to be know n
        collision_point_right = [
            int(
                self.rect.center[0]
                + math.cos(math.radians(self.angle + 18)) * length
            ),
            int(
                self.rect.center[1]
                - math.sin(math.radians(self.angle + 18)) * length
            ),
        ]

        collision_point_left = [
            int(
                self.rect.center[0]
                + math.cos(math.radians(self.angle - 18)) * length
            ),
            int(
                self.rect.center[1]
                - math.sin(math.radians(self.angle - 18)) * length
            ),
        ]

        try:
            if self.screen.get_at(collision_point_right) == pygame.Color(
                173, 255, 133, 255
            ) or self.screen.get_at(collision_point_left) == pygame.Color(
                173, 255, 133, 255
            ):
                self.alive = False

            pygame.draw.circle(
                self.screen, (0, 255, 255, 0), collision_point_right, 4
            )
            pygame.draw.circle(
                self.screen, (0, 255, 255, 0), collision_point_left, 4
            )

        except:
            self.alive = False

    def _did_rotate(self):
        """Checks whether the car rotates off the track and took wrong direction or not"""
        # TODO: The function implementation needs to be checked

        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(
            self.original_image, self.angle, 0.1
        )
        self.rect = self.image.get_rect(center=self.rect.center)

    def _update_radar(self, i: int, radar_angle: Union[int, float]) -> None:
        """The Car is made up of 6 radars. At every step this functions updates the radar to get the current direction and
        also updates the overall current status of the car.

        Args:
            i (int): The current index number
            radar_angle (Union[int, float]): The current angles in the radar.

        Returns:
            None
        """
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])
        try:
            while (
                not self.screen.get_at((x, y)) == pygame.Color(173, 255, 133, 255)
                and length < 200
            ):
                length += 1
                x = int(
                    self.rect.center[0]
                    + math.cos(math.radians(self.angle + radar_angle)) * length
                )
                y = int(
                    self.rect.center[1]
                    - math.sin(math.radians(self.angle + radar_angle)) * length
                )

            pygame.draw.line(
                self.screen, (255, 255, 255, 255), self.rect.center, (x, y), 1
            )
            pygame.draw.circle(self.screen, (0, 255, 0, 0), (x, y), 3)

            dist = int(
                math.sqrt(
                    math.pow(self.rect.center[0] - x, 2)
                    + math.pow(self.rect.center[1] - y, 2)
                )
            )
            self.radars[i] = dist
        except:
            self.alive = False

    def _drive(self):
        """Drives the car's center vector to the next state"""
        # TODO: Need to check why the value is 12 and nothing else
        self.rect.center += self.vel_vector * 12

    def draw(self) -> None:
        """
        Draws the car on the screen
        """
        self.screen.blit(self.track, (0, 0))
        self.screen.blit(self.image, self.rect.topleft)

    def timeTicking(self):
        self.clock.tick(self.FPS)

    def get_current_state(self, action: List[int]) -> Dict[str, Any]:
        """Returns the current state of the car. This states are determined the parameters of the Car mentioned below:
            - `is_alive` : This parameter determine whether the game is finished or not
            - `observation` : This parameter will return the current state of the car which are the radar's angular values and direction.
            - `direction` : This parameter will return the direction of the car.
            - `rotational_velocity` : This parameter will return the current rotational velocity of the car.
        Where each of the parameters are the keys of the dictionary returned by this function.

        Args:
            action (List[int]): The current action of the agent

        Returns:
            Dict[str, Any]: The current state of the car
        """
        if action[0] == 1:
            self.direction = -1
        elif action[1] == 1:
            self.direction = 1
        elif action[2] == 1:
            self.direction = 0
            self._drive()
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

        self._did_collide()
        self._did_rotate()

        for i, radar_angle in enumerate((-60, -30, 0, 30, 60)):
            self._update_radar(i, radar_angle)  # to be implemented

        if (
            self.radars[0] < 15
            and self.radars[4] < 15
            and self.radars[1] < 25
            and self.radars[2] < 25
            and self.radars[3] < 25
        ):
            self.alive = False
        else:
            self.alive = True

        self.params = {
            "isAlive": self.alive,
            "obs": self.radars,
            "dir": self.direction,
            "rotationVel": self.rotation_vel,
        }

        return self.params

    def play_step(self, action: List[int]) -> List[Any]:
        """Plays a single step of the game. This function is called by the agent during each step.

        Args:
            action (List[int]): The current action of the agent

        Returns:
            List[Any]: [current_reward, is_alive, overall_reward]
        """
        self.iterations += 1
        self._did_quit()
        self.draw()
        self.radars = [
            0,
            0,
            0,
            0,
            0,
        ]  # TODO: We have already initialised radars before, is there any need to initialize it here.
        self._drive()

        current_state_params = self.get_current_state(action=action)
        current_reward = self.reward_func(current_state_params)
        self.reward += current_reward
        
        if self.PYGAME_SCREEN_TYPE == "surface":
            pixel_data = pygame.surfarray.array3d(self.screen)
            return current_reward, not self.alive, self.reward, pixel_data
        else:
            return current_reward, not self.alive, self.reward 
