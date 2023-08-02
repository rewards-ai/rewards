# # DEPRICATED

# from rewards.environments.pygame.car_racer import Car, CarRacer, CarRacerConfig
# from rewards.models import LinearQNet
# from rewards.algorithms import DQN
# import pygame

# pygame.init()

# # TODO:
# # - Check if models are changing weights during training
# # - Create a train function either as different UTIL or within the evironment

# # Initialize cars, models and trainers
# cars = [Car(id="adam", radar_nums=5, show_radar=False), Car(id="eve", radar_nums=5, show_radar=False)]
# models = [LinearQNet([[cars[i].radar_nums, 64], [64, 3]]) for i in range(len(cars))]
# trainers = [DQN(models[i], epsilon=0.2, optimzer="adam") for i in range(len(cars))]

# # define a reward function
# def reward_function(params):
#     if params["is_alive"]:
#         return 1
#     return 0

# # create environment (CarRacer) with CarRacerConfig
# racer_config = CarRacerConfig(agents=cars, reward_function=reward_function, display_mode="surface")
# racer = CarRacer(racer_config)

# # Initialize trackers
# num_episodes = 1000
# total_scores = [0] * len(cars)
# records = [0] * len(cars)
# dones = [False] * len(cars)
# n_games = [0] * len(cars)

# n_test_after_steps = 100
# test_patient = 10000
        
        
# # define custom game loop
# while True:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#             exit()
            
#     # Loops over all agents (cars)
#     for i in range(len(cars)): 
        
#         # render the environment with agents
#         racer.render()
        
#         # train single step for a racer and save rewards
#         reward, dones[i], pixel_data = trainers[i].train_step(i, racer)
#         total_scores[i] += reward
        
#         # when done: reset; train from memory in batches; save model 
#         if dones[i]:
#             n_games[i] += 1
#             trainers[i].train_from_memory()
        
#             if total_scores[i] >= records[i]:
#                 records[i] = total_scores[i]
#                 trainers[i].model.save(model_name=cars[i].id)
                
#             total_scores[i] = 0
#             racer.agents[i].reset()
            
#             # Logging
#             import os
#             os.system('cls')
#             for j in range(len(cars)):
#                 print(cars[j].id, records[j], ":", "n_games:", n_games[j])
#                 print(trainers[j].epsilon)
#                 print("\n")
                
#             if n_games[i] % n_test_after_steps == 0:
#                 print("testing:", i)
#                 test_cars = [cars[i]]
#                 racer_config_temp = CarRacerConfig(agents=test_cars, reward_function=reward_function, display_mode="window")
#                 racer_temp = CarRacer(racer_config_temp)
#                 run_time = 0
#                 while run_time != test_patient:
#                     run_time += 1
#                     for event in pygame.event.get():
#                         if event.type == pygame.QUIT:
#                             pygame.quit()
                    
#                     if test_cars[0].isDisabled: break
#                     racer_temp.render()
#                     reward, is_done, pixel_data = trainers[i].predict_step(0, racer_temp)
#                     if is_done:
#                         racer_temp.agents[0].disable()
#                         break