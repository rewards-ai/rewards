from rewards_ai.Environments.Snake.SnakeTrainer.Agent import Agent
from rewards_ai.Environments.Snake.SnakeTrainer.Model.temp.Game import SnakeGameAI
import matplotlib.pyplot as plt


def rewards(props):
    if not props["isAlive"]:
        return -10
    if props["head"] == props["food"]:
        return 10
    return 1


plot_scores = []
plot_mean_scores = []
total_score = 0
record = 0
agent = Agent()
game = SnakeGameAI()
while True:
    # get old state
    state_old = agent.get_state(game)

    # get move
    final_move = agent.get_action(state_old)

    # perform move and get new state
    reward, done, score = game.play_step(final_move)
    print(reward)
    state_new = agent.get_state(game)

    # train short memory
    agent.train_short_memory(state_old, final_move, reward, state_new, done)

    # remember
    agent.remember(state_old, final_move, reward, state_new, done)

    if done:
        # train long memory, plot result
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
        plot(plot_scores, plot_mean_scores)
