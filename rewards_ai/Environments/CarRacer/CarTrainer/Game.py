import pygame
import math
from pathlib import Path
import time
import matplotlib.pyplot as plt

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


class Track:
    def __init__(self):
        self.track1_image = pygame.image.load(Path(__file__).parent.joinpath(r"Assets\track_test_7.png"))
        self.track1_checkpoints = [
            [(180, 144), (180, 44)],
            [(380, 144), (380, 44)],
            [(580, 144), (580, 44)],

            [(650, 180), (750, 180)],
            [(650, 350), (750, 350)],
            [(650, 500), (750, 500)],

            [(180, 553), (180, 653)],
            [(380, 553), (380, 653)],
            [(580, 553), (580, 653)],

            [(44, 180), (144, 180)],
            [(44, 350), (144, 350)],
            [(44, 500), (144, 500)],
        ]

        self.track1_rects = [
            pygame.Rect(180, 44, 2, 100),
            pygame.Rect(380, 44, 2, 100),
            pygame.Rect(580, 44, 2, 100),

            pygame.Rect(650, 180, 100, 2),
            pygame.Rect(650, 350, 100, 2),
            pygame.Rect(650, 500, 100, 2),

            pygame.Rect(180, 553, 2, 100),
            pygame.Rect(380, 553, 2, 100),
            pygame.Rect(580, 553, 2, 100),

            pygame.Rect(44, 180, 100, 2),
            pygame.Rect(44, 350, 100, 2),
            pygame.Rect(44, 500, 100, 2)
        ]

    def track1(self):
        return self.track1_image

    def track1_checkpoint(self, car):
        visited = []


class CarEnv:
    def __init__(self, reward_func, screen):
        self.screen = pygame.display.set_mode((800, 700))
        self.screen.fill((0, 0, 0))
        self.params = {
            "isAlive": None,
            "obs": None,
            "dir": None,
            "rotationVel": None
        }

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

        self.t = Track()
        self.track = self.t.track1_image

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
        if self.radars[0] < 15 and self.radars[4] < 15 and self.radars[1] < 25 and self.radars[2] < 25 and self.radars[
            3] < 25:
            self.alive = False
        else:
            self.alive = True

    # def draw_checkpoints(self):
    #     for rect in self.t.track1_rects:
    #         # pygame.draw.line(self.track, (0, 0, 0), points[0], points[1])
    #         pygame.draw.rect(self.screen, (255, 255, 255), rect)

    def draw(self):
        self.screen.blit(self.track, (0, 0))
        self.screen.blit(self.image, self.rect.topleft)
        # self.draw_checkpoints()
        # pygame.display.update()

    def timeTicking(self):
        self.clock.tick(self.FPS)

    def play_Step(self, action):
        self.iterations += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

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

        if self.radars[0] < 15 and self.radars[4] < 15 and self.radars[1] < 25 and self.radars[2] < 25 and self.radars[
            3] < 25:
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

    def train(self, MODE, CONTROL_SPEED, TRAIN_SPEED, agent):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0

        # ma = []
        # ma_x = []
        # ma_mean = []
        # ma_total = 0
        # ma_interval = 10

        record = 0
        while True:
            pygame.display.update()
            x = pygame.surfarray.array3d(self.screen)
            print(x)
            if MODE == "human":
                time.sleep(CONTROL_SPEED)
                self.play_human()
            else:
                self.FPS = TRAIN_SPEED
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
                    # plot(score, plot_scores, total_score, plot_mean_scores, agent)
                    plot_scores.append(score)
                    total_score += score
                    mean_score = total_score / agent.n_games
                    plot_mean_scores.append(mean_score)

                    # if agent.n_games >= ma_interval and agent.n_games % ma_interval == 0:
                    #     print("here")
                    #     ma_x.append(agent.n_games)
                    #     ma.append(score)
                    #     ma_total += score
                    #     ma_m = ma_total / (agent.n_games // ma_interval)
                    #     ma_mean.append(ma_m)

                    plt.clf()
                    plt.title('Training...')
                    plt.xlabel('Number of Games')
                    plt.ylabel('Score')
                    plt.plot(plot_scores)
                    plt.plot(plot_mean_scores)
                    # plt.plot(ma_x, ma_mean)
                    plt.ylim(ymin=0)
                    plt.text(len(plot_scores) - 1, plot_scores[-1], str(plot_scores[-1]))
                    plt.text(len(plot_mean_scores) - 1, plot_mean_scores[-1], str(plot_mean_scores[-1]))
                    plt.show(block=False)
                    plt.pause(.1)
