import pygame
import math
import os
from pathlib import Path

pygame.init()


def convert_points(point):
    return int(point[0]), int(point[1])


class CarEnv:
    def __init__(self, reward_func):
        self.screen = pygame.display.set_mode((800, 700))
        self.screen.fill((0, 0, 0))

        self.original_image = None
        self.angle = None
        self.image = None
        self.rect = None
        self.vel_vector = None
        self.rotation_vel = None
        self.direction = None
        self.alive = None
        self.radars = [0, 0, 0, 0, 0]
        self.reward = 0
        self.drive_factor = 12
        self.clock = pygame.time.Clock()

        self.FPS = 15
        self.iterations = 0
        self.reward_func = reward_func
        self.track = pygame.image.load(Path(__file__).parent.joinpath(r"Assets\track_test_7.png"))

        self.initialize()

    def initialize(self):
        car_Scale = 500
        self.screen = pygame.display.set_mode((800, 700))
        self.original_image = pygame.image.load(Path(__file__).parent.joinpath(r"Assets\car.png"))
        self.original_image = pygame.transform.scale(self.original_image, (car_Scale, car_Scale))
        self.angle = 0
        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=(200, 100))
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.rotation_vel = 15
        # self.vel_vector = pygame.math.Vector2(1.6, 0)
        # self.rotation_vel = 10
        # self.vel_vector = pygame.math.Vector2(2.4, 0)
        # self.rotation_vel = 7
        self.direction = 0
        self.alive = True
        self.reward = 0
        self.radars = [0, 0, 0, 0, 0]

        self.drive_factor = 12

        return [self.radars]

    def check_failed(self):
        if self.radars[0] < 15 and self.radars[4] < 15 and self.radars[1] < 25 and self.radars[2] < 25 and self.radars[3] < 25:
            self.alive = False
        else:
            self.alive = True

    def draw(self):
        self.screen.blit(self.track, (0, 0))
        self.screen.blit(self.image, self.rect.topleft)
        # pygame.display.update()

    def timeTicking(self):
        self.clock.tick(self.FPS)

    def play_Step(self, action):
        self.iterations += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self.draw()
        self.radars = [0, 0, 0, 0, 0]
        self.drive()
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
        self.rotate()

        self.collision()
        for i, radar_angle in enumerate((-60, -30, 0, 30, 60)):
            self.radar(i, radar_angle)

        if self.radars[0] < 15 and self.radars[4] < 15 and self.radars[1] < 25 and self.radars[2] < 25 and self.radars[3] < 25:
            self.alive = False
        else:
            self.alive = True

        reward = self.reward_func({
            "isAlive": self.alive,
            "obs": self.radars,
            "dir": self.direction
        })

        self.reward += reward
        return reward, not self.alive, self.reward

    def drive(self):
        self.rect.center += self.vel_vector * 12

    def collision(self):
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
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, i, radar_angle):
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
