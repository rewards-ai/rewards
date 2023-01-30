from agent import Agent
from game import SnakeGameAI, Direction
import matplotlib.pyplot as plt
import pygame
import time

MODE = "human"

plot_scores = []
plot_mean_scores = []
total_score = 0
record = 0
agent = Agent()
game = SnakeGameAI()
prev_dir = None
while True:
    state_old = agent.get_state(game)
    final_move = agent.get_action(state_old)

    reward, done, score = game.play_step(final_move)
    state_new = agent.get_state(game)

    agent.train_short_memory(state_old, final_move, reward, state_new, done)
    agent.remember(state_old, final_move, reward, state_new, done)

    if done:
        game.reset()
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