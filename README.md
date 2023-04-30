# **rewards**

#### Create custom environments and train custom agents in few lines of code.

Getting started with RL is quite easy now a days. The workflow stays almost same. You create your environment. This environment is either used for from Open AI's `gym` or we make custom environment using `pygame` and `unity`. After environment creation we go for making deep RL agents, by creating our model using `tensorflow` or `pytorch` etc. 
So the bottleneck mostly lies in the environment creation, integrating the environment with different libraries like `gymnasium` to make agents around it and finding that best reward function. It becomes very hectic to manage all these experimentation process all by yourself. 
Introducing **rewards**, a low code RL training and experimentation platform powered by rewards.ai, rewards lets us to do those in some few lines of code. Manage all your RL experimentation and integration code in just few lines of code.

### Getting started

Oh that's very easy. First install rewards:

```bash
pip install --upgrade rewards
```

This should install the latest version of rewards. After this in few lines of code you can get started by ceating your first experiment.

> rewards currently only support it's own racing environment. Support for more environments, custom environment and gym will come in next version.

First import some required modules from `rewards` and `rewards_envs`.

```python
from rewards import LinearQNet, QTrainer
from rewards_envs import CarConfig, CarGame

# import some additional modules too 
import cv2 
import matplotlib.pyplot as plt 
```

> Current version of rewards already assumes that it is working on it's car-race environment. In coming version we will provide support for custom env integration.

Once everything is imported then in order to build our custom environment, you have to simply write these few lines of code.

```python
# First create the configurations to create the environment 
# configurations helps to maintain multiple experiments 

env_config = CarConfig(
    render_mode = "rgb_array", 
    car_fps=30, screen_size = (1000, 700)
)

# create the game 
env = CarGame(
    mode = "training", 
    track_num = 1, 
    reward_function = reward_function, 
    config = env_config
)
```

If you see when you are initializing the environment, there is a parameter called `reward_function` : `Callable`. This is a function that you have to define based on the given car's properties. Below is a sample reward function that works best for this environment.

```python
def reward_function(props) -> int:
    reward = 0
    if props["is_alive"]:
        reward = 1
    obs = props["observation"]
    if obs[0] < obs[-1] and props["direction"] == -1:
        reward += 1
        if props["rotational_velocity"] == 7 or props["rotational_velocity"] == 10:
            reward += 1
    elif obs[0] > obs[-1] and props["direction"] == 1:
        reward += 1
        if props["rotational_velocity"] == 7 or props["rotational_velocity"] == 10:
            reward += 1
    else:
        reward += 0
        if props["rotational_velocity"] == 15:
            reward += 1
    return reward
```

The agent (here the car) has some following properties named under the dictionary `props`. Here the name and the explaination of all the properties.

- `is_alive` : This states, whether the car is alive or not
  
- `observation` : Observation is a array of 5 float values, which are the radars of the car.
  
- `direction` : Direction provides the current action taken by the car.
  
- `rotational_velocity` : The rotational velocity of the car.
  

The properties of the car are determined during the process of creation of the game. If you want to create a custom environment, then you can define your agent's properties there. The propeties must be those, which determines whether or how much an agent is gonna win/loose the game.

After setting up the reward function and the game environment, then build the model abd agent. Its very simple, just few lines of code. The model you are building supports both `LinearQNet` or a custom pytorch model too.

Note: In this case, the input and the output neurons are fixed for the environment. The input neurons are the radars of the car and the output neurons are the action probabilities of the car, that determines which action to choose.

```python
# Create a very basic model 

model = LinearQNet(layers_conf=[[5, 128], [128, 3]])

# create an agent 

agent = QTrainer(
    lr = 0.001, 
    gamma = 0.99, 
    epsilon = 0.10, 
    model = model, 
    loss = "mse", 
    optimizer = "adam", 
    checkpoint_folder_path = None, 
    model_name = "model_3.pth"
)
```

As a last step, create a training loop. It's kinda similar to the PyTorch training loop.

```python
# write a small training loop 

# Initialize two list to plot the metrics 
plot_scores, plot_mean_scores = [], []
total_score, record = 0, 0 
num_episodes = 600

for episode in range(1, num_episodes + 1):
    _, done, score, pixel_data = agent.train_step(game)
    game.timeTicking() 

    # show the game frame 
    cv2.imshow('Frame', pixel_data)
    if cv2.waitKey(30) & 0xFF == ord('q') : break 
    
    if done:
        # initialize the game 
        game.initialize() 
        agent.n_games += 1 
        
        # Make the agent to remmember about it's state 
        agent.train_long_memory() 
        
        # If the current episode score is greater than record then save the model
        if score > record:
            record = score 
            agent.model.save(model_name = "modelv1.pth")
            print(f"with a record of: {record}")
            
        # print and plot all the metrics of that episode 
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
```

And wallah, you have successfully trained your first agent using rewards.

**what other things rewards provides**?

- Easy experimentation and integration management in just few lines of code.
- Integration with **[rewards-platform](https://github.com/rewards-ai/training-platform)** .If you did' check out, think it as the open source alternative of Amazon deep racer.
- Beginner friendly documentation focussed on learning reinforcement learning.

---

## **Setting up the project locally**

Setting up rewards is very easy. All you have to do is to first create a virtual environment. Creating a virtual environment is very easy:
**`[LINUX]`**

```bash
$ virtualenv .rewards
$ source .rewards/bin/activate
```

**`[WINDOWS]`**

```bash
virtualenv .rewards
.\venv\Scripts\Activate
```

After this clone the repository. To clone the repo and move inside the directory, just type the command:

```bash
$ git clone https://github.com/rewards-ai/rewards-SDK.git
$ cd rewards-SDK
```

```bash
pip install poetry
```

After this install all the dependencies by:

```bash
poetry install
```

That's it. After this latest version of `rewards` get's installed and you can work on the top of it.

## **Contributing**

Both `rewards` and `rewards_envs` are undergoing through some heavy developement. Being a open source projects we are open for contributions. Write now due to lack of docs, we are unable to come down with some guidelines. We will be doing that very soon. Till then please star this project and play with our sdk. If you find some bugs or need a new feature, then please create a new issue under the issues tab. Right now we expect contribution in terms of adding:

- Tests : Provide test scripts for testing our current rewards package.
- Feature requests : Feel free to request more features in the issues tab.
- Documentation : We are a beginner first organisation. Our documentation not only focuses on how to use rewards in their RL workflow but also to learn and get started with RL at the same time. So if you are interested in contributing that, please feel free to do so.
  
  ### **How to build the docs**
  
  rewards uses `docusaurus` to create the docs. Go inside the `docs/` folder and build the docs by installing the dependencies and start the docs server.
  
  ```bash
  $ npm i
  $ npm run start
  ```
  

---

## **Roadmap**

rewards is under heavy developement and updates get rolled very frequently. We are currently focussed on building our sdk such that it supports our other projects that includes `rewards-api` and `rewards-platform`. We want to make rewards as a general repository for RL research and RL education. Most of the RL research are heavily dependent on the environment. After environment creation, practicioners either face lot of issues wraping that environment around `gymnasium` 's custom environment wrapper or create everything of their own. We at rewards want to solve this issue. Through `rewards-sdk` and `rewards_envs` user must be able to create custom environment made using Pygame, Unity or any other engine and integrate it and start/organize RL research in no time.