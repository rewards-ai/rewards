from rewards.environments.pygame.car_racer import Car, CarRacer, CarRacerConfig
from rewards.models import LinearQNet
from rewards.algorithms import DQN
import pygame

pygame.init()

# Initialize cars, models and trainers
cars = [Car(id="titu", radar_nums=5, show_radar=False)]
models = [LinearQNet([[cars[i].radar_nums, 64], [64, 3]]) for i in range(len(cars))]
trainers = [DQN(models[i], epsilon=0.2, optimzer="adam") for i in range(len(cars))]

# define a reward function
def reward_function(params):
    if params["is_alive"]:
        return 1
    return 0

# create environment (CarRacer) with CarRacerConfig
racer_config = CarRacerConfig(agents=cars, reward_function=reward_function)
racer = CarRacer(racer_config)

# Initialize trackers
num_episodes = 1000
total_scores = [0] * len(cars)
records = [0] * len(cars)
dones = [False] * len(cars)
n_games = [0] * len(cars)
        
        
# define custom game loop
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
            os.system('cls')
            for j in range(len(cars)):
                print(cars[j].id, records[j], ":", "n_games:", n_games[j])
                print(trainers[j].epsilon)
                print("\n")