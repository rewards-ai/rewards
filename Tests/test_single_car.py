import matplotlib.pyplot as plt
from rewards_ai.Environments.CarRacer.CarTrainer import Game, Agent
from rewards_ai.Model.DQN import Linear_QNet
import pygame

plt.ion()

MODE = "training"
load_last_checkpoint = False
CONTROL_SPEED = 0.05
TRAIN_SPEED = 100
screen = pygame.display.set_mode((800, 700))


# write reward func
def reward_func(props):
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


# create model arch
linear_net = Linear_QNet([5, 128, 3])

# initialize game and agent
agent = Agent.Agent(linear_net, load_last_checkpoint)
game = Game.CarEnv(reward_func, screen)

# training loop
game.train(
    MODE, CONTROL_SPEED, TRAIN_SPEED, agent
)
