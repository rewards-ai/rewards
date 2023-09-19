from rewards.models import LinearQNet
from rewards.algorithms import DQN
from rewards.environments.pygame.car_racer import Track, Car, CarRacer, CarRacerConfig
import pygame

pygame.init()

# Constants
WIDTH, HEIGHT = 800, 700
track = Track(size=(WIDTH, HEIGHT), generated=True, seed=-1)

cars = [
    Car(id="agent-1", radar_nums=5, show_radar=True, angle=track.direction, center=track.start_point),
    Car(id="agent-2", radar_nums=5, show_radar=True, angle=track.direction, center=track.start_point)
]

models = [LinearQNet([[cars[i].radar_nums, 64], [64, 3]]) for i in range(len(cars))]
trainers = [DQN(models[i], epsilon=0.2, optimzer="adam", epsilon_decay_rate=0.0001) for i in range(len(cars))]

# Create a Pygame window
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# define a reward function
def reward_function(params):
    if params["is_alive"]:
        return 1
    return 0

# create environment (CarRacer) with CarRacerConfig
racer_config = CarRacerConfig(
    agents=cars, 
    reward_function=reward_function,
    track=track,
    FPS=50
)
racer = CarRacer(racer_config)

# Initialize trackers
num_episodes = 1000
total_scores = [0] * len(cars)
records = [0] * len(cars)
dones = [False] * len(cars)
n_games = [0] * len(cars)

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
            
    # Loops over all agents (cars)
    for i in range(len(cars)): 
        
        # render the environment with agents
        racer.render()
        
        # train single step for a racer and save rewards
        reward, dones[i], pixel_data = trainers[i].train_step(i, racer)
        total_scores[i] += reward
        
        # when done: reset; train from memory in batches; save model 
        if dones[i]:
            n_games[i] += 1
            trainers[i].train_from_memory()
        
            if total_scores[i] >= records[i]:
                records[i] = total_scores[i]
                trainers[i].model.save(model_name=cars[i].id)
                
            total_scores[i] = 0
            racer.agents[i].reset()
            
            # Logging
            import os
            # os.system('cls')
            print(track.direction)
            for j in range(len(cars)):
                print(cars[j].id, records[j], ":", "n_games:", n_games[j])
                print("epsilon", trainers[j].epsilon)
                print("\n")
