from rewards_ai.Environments.CarRacer.CarTester.Game import Car, GameController
from rewards_ai.Model.DQN import Linear_QNet, QTrainer
import matplotlib.pyplot as plt
import pygame
import time
import torch

plt.ion()

N_CARS = 1

plot_scores = []
plot_mean_scores = []
total_score = 0
record = 0

CARS = [Car()] * N_CARS
MODELS = [Linear_QNet([5, 10, 3])] * N_CARS
MODEL_PATHS = [r".\model\modelCAR1.pth"]

for i, model in enumerate(MODELS):
    model.load_state_dict(torch.load(MODEL_PATHS[i]))
    model.eval()

game = GameController(CARS)

while True:
    time.sleep(0.05)
    pygame.display.update()
    action = [[0]*5]*N_CARS

    res = [MODELS[i](torch.tensor(CARS[i].radars, dtype=torch.float)) for i in range(N_CARS)]

    for i, act in enumerate(action):
        print(torch.argmax(res[i]).item(), i)
        action[i][torch.argmax(res[i]).item()] = 1

    game.play_Step(action)
    if not game.alive:
        game.initialize()
