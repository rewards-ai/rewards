import torch
import random
import numpy as np
from collections import deque
from rewards_ai.Model.DQN import QTrainer

MAX_MEMORY = 100000
BATCH_SIZE = 1000


class Agent:
    def __init__(self, model, lr=0.001, epsilon=0.25, gamma=0.9):
        self.n_games = 0
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = model
        self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma)

    def get_state(self, game):
        state = game.radars
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        return self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 30
        final_move = [0, 0, 0]
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def train_step(self, game):
        state_old = self.get_state(game)
        final_move = self.get_action(state_old)
        reward, done, score = game.play_Step(final_move)
        state_new = self.get_state(game)
        self.train_short_memory(state_old, final_move, reward, state_new, done)
        self.remember(state_old, final_move, reward, state_new, done)

        return reward, done, score

