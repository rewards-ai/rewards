from rewards.models import LinearQNet
from rewards.agent import Agent
from rewards.envs.car import CarGame
import pygame
import matplotlib.pyplot as plt

def plot(score, plot_scores, total_score, plot_mean_scores, agent):
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


plot_scores = []
plot_mean_scores = []
total_score = 0
record = 0

def reward_func(props):
    reward = 0
    if props["isAlive"]:
        reward = 1
    obs = props["obs"]
    if obs[0] < obs[-1] and props["dir"] == -1:
        reward += 1
    elif obs[0] > obs[-1] and props["dir"] == 1:
        reward += 1
    else:
        reward += 0
    return reward

linear_net = LinearQNet([[5, 9], [9, 3]])
agent = Agent(linear_net)
game = CarGame(
    track_num=0,
    reward_func=reward_func
)

while True:
    pygame.display.update()
    print("test", game.track_image_path)
    reward, done, score = agent.train_step(game)
    game.timeTicking()

    if done:
        game.initialize()
        agent.n_games += 1
        if agent.play_trained:
            print('Game', agent.n_games, 'Score', score)
        else:
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
        plot(score, plot_scores, total_score, plot_mean_scores, agent)