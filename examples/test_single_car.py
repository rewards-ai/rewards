import matplotlib.pyplot as plt
from rewards_ai.Environments.CarRacer.CarTrainer import Game, Agent
from rewards_ai.Model.DQN import Linear_QNet
import pygame
import time

plt.ion()

MODE = "training"
plot_scores = []
plot_mean_scores = []
total_score = 0
record = 0


# write reward func
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


# create model arch
linear_net = Linear_QNet([5, 10, 3])

# initialize game and agent
agent = Agent.Agent(linear_net)
game = Game.CarEnv(reward_func)

# initialize running parameters
CONTROL_SPEED = 0.05
TRAIN_SPEED = 100

while True:
    pygame.display.update()
    if MODE == "human":
        time.sleep(CONTROL_SPEED)
        game.play_human()
    else:
        game.FPS = TRAIN_SPEED
        reward, done, score = agent.train_step(game)
        game.timeTicking()

        if done:
            game.initialize()
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