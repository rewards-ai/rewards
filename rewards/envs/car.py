
import os
import math
import pygame
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class CarConfig:
    def __init__(
            self,
            display_type: Optional[str] = None,
            screen_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Car environment configurations where the agent will be trained. 

        Args:
            param:(display_type) : Optional[str] : Whether to pop up a window or return the screen as a stream 
            There are two options available ("display", "surface"). 
            Using `display` will create a pygame window. This is used when SDK is used directly for training. 
            Using `surface` will generate a stream of pygame display. This can be used when the pygame screen is used 
            for recording purposes or some other. 

            param:(screen_size) : Optional[Tuple[int, int]] : The size of the screen. The default screen is taken 
            as (800, 700)

        TODO: 
        -----
            - Use Hydra for default configuration management
            - Implement support for custom environments tracks 
            - Cusom Car configuration support including radars (can be implemented)
            - Is FPS is same as CAR_SPEED ? 
            - What is CAR_ANGLE?
            - Provide the documentation for car skeleton configuration and what each of them does 
        """

        # Global Path
        self.PARENT_PATH: str = Path(__file__).parent.parent

        # Car Assets path
        self.ASSET_PATH: str = str(
            os.path.join(self.PARENT_PATH, "assets/CarRace")
        )

        # Car default configuration (remains unchanged)

        self.CAR_SCALE: int = 500
        self.CAR_IMAGE: str = "car.png"
        self.DRIVE_FACTOR: int = 12
        self.CAR_FPS: int = 15
        self.CAR_ANGLE: int = 0

        # Training and evaluation environment paths

        self.TRAINING_CAR_TRACKS: Dict[int, str] = {
            1: "track-1.png",
            2: "track-2.png",
            3: "track-3.png",
        }

        self.EVALUATION_CAR_TRACKS: Dict[int, str] = {
            1: "track-1.png"
        }

        # Car Skeleton configuration

        self.CAR_RECT_SIZE: Tuple[int, int] = (200, 50)
        self.CAR_VELOCITY_VECTOR: Tuple[float, float] = (0.8, 0.0)
        self.CAR_ROTATION_VELOCITY: Union[int, float] = 15
        self.CAR_DIRECTION: Union[int, float] = 0
        self.CAR_IS_ALIVE: bool = True
        self.CAR_REWARD: int = (0,)
        self.CAR_RADAR: List[Union[int, float]] = [0, 0, 0, 0, 0]

        # Pygame display screen configuration

        self.PYGAME_SCREEN_TYPE: str = "display" if display_type is None else display_type
        self.SCREEN_SIZE: Tuple[int, int] = (
            800, 700) if screen_size is None else screen_size


class CarGame(CarConfig):
    def __init__(
            self,
            track_num: int,
            mode: str,
            reward_function: Optional[Callable] = None,
            display_type: Optional[str] = None,
            screen_size: Optional[Tuple[int, int]] = None) -> None:
        """
        Car Game Environment 
        --------------------
        Meet reward's car-game environment. There are mainly 4 main types of operations under which this 
        environment operates. Those are as follows:

            - `isAlive` : This parameter determine whether the game is finished or not
            - `obs` : This represents the car's current observation. 
            - `dir` : This represents the car's current direction 
            - `rotationVel` : This represents the current rotational velocity of the car.

        The above four parameter are very much important as that will be used for writing custom writing 
        reward function by the user. 

        Args:
            param:(track_num) : int : Which training track will be used for training or validation. 
            For training there are three tracks options supported (1,2,3) and for evaluation only (1)
            track is supported

            param:(mode) : str: There are two options: (training / evaluation). Based on this the 
            car's environment will be choosen. 

            param:(reward_function) : Optional[Callable], optional : reward function on which the car
            will do it's scoring Defaults to None. This parameter expects a function with argument props
            where props is of type Dict containing the above parameters (isAlive, obs, dir, rotationVel)

            param:(display_type) : Optional[str], optional: The type of the display that is to be used. There 
            are two options available ("surface", "display"). 
            Using `display` will create a pygame window. This is used when SDK is used directly for training. 
            Using `surface` will generate a stream of pygame display. This can be used when the pygame screen 
            is used for recording purposes or some other. 

            param:(screen_size) : Optional[Tuple[int, int]] : The size of the screen. The default screen is taken 
            as (800, 700)
        """

        if mode != "training" and mode != "evaluation":
            raise ValueError("mode must be either `training` or `evaluation`")

        # print(display_type == "surface")
        # if display_type != "display" and mode != "surface":
        #     raise ValueError("display must be either `display` or `surface`")

        if mode == "training" and (track_num < 1 or track_num > 3):
            raise ValueError(
                "`training` mode only supports three types of environment: Options: (1,2,3)")

        if mode  == "evaluation" and track_num != 1:
            raise ValueError(
                "`evaluation` mode only supports one type of environment: Options: (1)")

        super(CarGame, self).__init__(
            display_type=display_type, screen_size=screen_size)
        
        self.mode = mode
        
        # Loading car and track path
        self.track_options = self.TRAINING_CAR_TRACKS if mode == "training" else self.EVALUATION_CAR_TRACKS
        self.track_image_path = os.path.join(
            self.ASSET_PATH, mode, self.track_options[track_num]
        ).replace(os.sep, '/')
        self.car_image_path = os.path.join(self.ASSET_PATH, self.CAR_IMAGE)

        # building track and car path
        self.track_image = pygame.image.load(self.track_image_path)
        self.car_image = pygame.transform.scale(
            pygame.image.load(
                self.car_image_path), (self.CAR_SCALE, self.CAR_SCALE),
        )

        # Building PyGame screen
        self.screen = pygame.display.set_mode(
            self.SCREEN_SIZE) if self.PYGAME_SCREEN_TYPE == "display" else pygame.Surface(self.SCREEN_SIZE)
        
        self.screen.fill((0, 0, 0))

        # All the car configurations
        self.angle = self.CAR_ANGLE
        self.original_image = self.car_image
       
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.CAR_RECT_SIZE)
        self.vel_vector = pygame.math.Vector2(self.CAR_VELOCITY_VECTOR)
        self.rotation_vel = self.CAR_ROTATION_VELOCITY
        
        self.direction = self.CAR_DIRECTION
        self.drive_factor = self.DRIVE_FACTOR
        self.alive = self.CAR_IS_ALIVE
        self.radars = self.CAR_RADAR
        self.reward = 0

        # Additional configuration
        self.clock = pygame.time.Clock()
        self.track = self.track_image
        self.iterations = 0
        self.FPS = self.CAR_FPS

        # Initial parameter for reward function
        self.params = {
            "is_alive": None,
            "observation": None,
            "direction": None,
            "rotational_velocity": None,
        }
        self.reward_function = self._default_reward_function if reward_function is None else reward_function

    def _default_reward_function(self, props: Dict[str, Any]) -> Union[int, float]:
        """
        Default reward function that will be used if no custom reward function is provided. This is a very simple 
        reward function where if the agent is alive then the reward will be 1 else 0. 

        Args:
            props (Dict[str, Any]): Properties of the agent to see in the environment. Here are the properties:
                - `isAlive` : This parameter determine whether the game is finished or not
                - `obs` : This represents the car's current observation. 
                - `dir` : This represents the car's current direction 
                - `rotationVel` : This represents the current rotational velocity of the car.

        Returns:
            Union[int, float]: reward that the agent got. 
        """
        if props["isAlive"]:
            return 1
        return 0

    def initialize(self) -> None:
        """
        Initializes the car environment with all the default properties
        """
        self.angle = self.CAR_ANGLE
        self.original_image = self.car_image
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.CAR_RECT_SIZE)
        self.vel_vector = pygame.math.Vector2(self.CAR_VELOCITY_VECTOR)
        self.rotation_vel = self.CAR_ROTATION_VELOCITY
        self.direction = self.CAR_DIRECTION
        self.drive_factor = self.DRIVE_FACTOR
        self.alive = self.CAR_IS_ALIVE
        self.radars = self.CAR_RADAR
        self.reward = 0

    def _did_quit(self):
        """
        Quits the game when user presses the 'quit'/'close' key.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def _did_collide(self):
        """
        Checks the status whether the car collied or not. If the car collides, 
        then `isAlive` is False and game terminates.

        TODO: 
        -----
        - This function needs to be checked
        """

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
        """
        Checks whether the car rotates off the track and took wrong direction or not

        TODO: 
        -----
        The function implementation needs to be checked
        """

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
        """
        The Car is made up of 6 radars. At every step this functions updates the radar to get 
        the current direction and also updates the overall current status of the car.

        Args:
            i (int): The current index number
            radar_angle (Union[int, float]): The current angles in the radar.
        """
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])
        try:
            while (
                not self.screen.get_at(
                    (x, y)) == pygame.Color(173, 255, 133, 255)
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
        """
        Drives the car's center vector to the next state
        TODO
        ----
        - Need to check why the value is 12 and nothing else
        """
        self.rect.center += self.vel_vector * self.drive_factor

    def draw(self) -> None:
        """
        Draws the car on the screen
        """
        self.screen.blit(self.track, (0, 0))
        self.screen.blit(self.image, self.rect.topleft)

    def timeTicking(self):
        self.clock.tick(self.FPS)

    def get_current_state(self, action: List[int]) -> Dict[str, Any]:
        """
        Returns the current state of the car. This states are determined the parameters 
        of the Car mentioned below:
            - `isAlive` : This parameter determine whether the game is finished or not
            - `obs` : This represents the car's current observation. 
            - `dir` : This represents the car's current direction 
            - `rotationVel` : This represents the current rotational velocity of the car.
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

    def play_step(self, action: List[str]) -> List[Any]:
        """
        Plays a single step of the game. This function is called by the agent during each step.

        Args:
            action (List[int]): The current action of the agent

        Returns:
            List[Any]: [current_reward, is_alive, overall_reward]
        """
        self.iterations += 1
        
        if self.PYGAME_SCREEN_TYPE == "display":
            self._did_quit()
        self.draw()
        self.radars = [0, 0, 0, 0, 0]
        self._drive()

        current_state_params = self.get_current_state(action=action)
        current_reward = self.reward_function(current_state_params)
        self.reward += current_reward

        if self.PYGAME_SCREEN_TYPE == "surface":
            pixel_data = pygame.surfarray.array3d(self.screen)
            return current_reward, not self.alive, self.reward, pixel_data
        else:
            return current_reward, not self.alive, self.reward
