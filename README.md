# **rewards**

#### Create custom environments and train custom agents in few lines of code.

Getting started with RL is quite easy now a days. The workflow stays almost same. You create your environment. This environment is either used for from Open AI's `gym` or we make custom environment using `pygame` and `unity`. After environment creation we go for making deep RL agents, by creating our model using `tensorflow` or `pytorch` etc. 

So the bottleneck mostly lies in the environment creation, integrating the environment with different libraries like `gym` to make agents around it and finding that best reward function. It becomes very hectic to manage all these experimentation process all by yourself. 

Introducing **rewards**, a low code RL training and experimentation platform powered by rewards.ai, rewards lets us to do those in some few lines of code. Manage all your RL experimentation and integration code in just few lines of code. 

**what other things rewards provides**?

- Easy experimentation and integration management in just few lines of code. 

- Integration with **[rewards-platform]([GitHub - rewards-ai/training-platform](https://github.com/rewards-ai/training-platform))** .If you did' check out, think it as the open source alternative of Amazon deep racer. 

- Beginner friendly documentation focussed on learning reinforcement learning.



## **Getting started**

Oh that's very easy. First install rewards:

```bash
pip install --upgrade rewards
```

This should install the latest version of rewards. After this in few lines of code you can get started by ceating your first experiment. 


> rewards currently only support it's own racing environment. Support for more environments, custom environment and gym will come in next version.



First import our `workflow` module. The workflow module will create setup an experimentation setup for a specific RL environment. This mainly stores the **configuration**, **metrics**, **reward functions**, **model weights**,**environment**. Once everything is stored you can easily integrate experimentation tracking services **weights and biases** or **neptune.** You can check out those tutorials in our documentation. 

Every experiment expects some sets of configuration. A typical configuration in rewards looks like this:

```python
configs = workflow.WorkFlowConfigurations(
    MODE="training", 
    LAYER_CONFIG=[[5, 16], [16, 3]], 
    CHECKPOINT_FOLDER_PATH = None, 
    CHECKPOINT_MODEL_NAME="model.pth", 
    NUM_EPISODES=1000, 
    CAR_SPEED=20, 
    REWARD_FUNCTION=reward_func, 
    PYGAME_WINDOW_TYPE="display", 
    ENVIRONMENT_WORLD=1,
    SCREEN_SIZE=(1000, 700), 
    EPSILON=0.99, 
    GAMMA=0.9, 
    LR=0.001
)
```

> Current version of rewards already assumes that it is working on it's car-race environment. In coming version we will provide support for custom env integration. 

Here is the meaning of each of the configuration parameters:

**`Mode`**:  This states in which mode our experiment will run on. There are two types: `training` and `evaluation`. In training, the agent can run for multiple episodes and goes for the deep learning optimization. In `evaluation` mode, there is no further optimization and it runs for a single episode. 

**`LAYER_CONFIG`**: Current version of rewards supports a simple neural network builder. We already assume that the input and output layer will have no of neurons of 5 and 3 respectively. Rest every number of neurons and layers can be configured.

**`CHECKPOINT_FOLDER_PATH`**:  Specifies where the model checkpoints will be saved. By default it is None. If changed then it will save the model on that specific path.

**`CHECKPOINT_MODEL_NAME`**: The name of the model that is to be saved. By default the name is setup to be as `model.pth`. 

**`NUM_EPISODES`**: The number of episodes the model should be trained. 

**`CAR_SPEED`**: The speed of the car while traning. The range of the speed lies from 1 to 100. It is recommended to train the agent with lower speed initially and gradually increase the speed for more generalised agent. 

**`REWARD_FUNCTION`**: Reward function is a function that powers reinforcement learning agents learn about the environment. Better the reward function better the learning. A sample reward function should look this:

```python
def reward_func(props) -> int:
    reward = 0
    if props["isAlive"]:
        reward = 1
    obs = props["obs"]
    if obs[0] < obs[-1] and props["dir"] == -1:
        reward += 1
        if props["rotationVel"] == 7 or props["rotationVel"] == 10:
            reward += 1
    elif obs[0] > obs[-1] and props["dir"] == 1:
        reward += 1
        if props["rotationVel"] == 7 or props["rotationVel"] == 10:
            reward += 1
    else:
        reward += 0
        if props["rotationVel"] == 15:
            reward += 1
    return reward
```

Here `props` represents properties. This includes the properties of the agent. Here is the car. The properties include `isAlive` (whether the game finished or not), `obs` (current radar observation), `dir` (current car direction) `rotationVel` (Car's rotational velocity)

**`PYGAME_WINDOW_TYPE`**: There are two types. `display` takes out the pygame window while training the agent. Where as `surface` streams the pygame window streams as image to show to somewhere else. Usecases include to integrate to other platforms or saves as video. 

**`ENVIRONMENT_WORLD`**: The type of the track that is to be choosen. For `training` there are three options 1/2/3. For evaluation there is one option: 1. 

**`SCREEN_SIZE`**: The size of the pygame window screen. 

**`GAMMA`**: A discount factor that determines the importance of future rewards in the RL algorithm.

**`EPSILON`**: A probability value used in the epsilon-greedy strategy to balance exploration and exploitation.

**`LR`**: Learning rate, A hyperparameter that controls the step size in updating the weights of a neural network during training.. 



After configuring all the configurations and writing the reward function we call the `rewads.workflow.RLWorkFlow` module that will take all the configurations and start the experiments. 

```python
flow = workflow.RLWorkFlow(configs)
flow.run_episodes()
```

****

## **Setting up the project locally**

Setting up rewards is very easy. All you have to do is to first create a virtual environment. Creating a virtual environment is very easy:

**`[LINUX]`**

```bash
```bash
$ virtualenv .rewards
$ source .rewards/bin/activate
```
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

You should have poetry to install all the dependencies. If poetry is not installed then install it by: 

```bash
pip install poetry
```

After this install all the dependencies by:

```bash
poetry install
```

That's it. After this latest version of `rewards` get's installed and you can work on the top of it. 

---

## **Contributing**

We will be very happy to expand the rewards community as soon as possible. Right now we expect contribution in terms of adding:

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

rewards is under heavy developement and updates get rolled very frequently. We are currently focussed on building our sdk such that it supports our other projects that includes `rewards-api` and `rewards-platform`. But here are the major changes and updates we are looking forward. 

- [ ] Introducing more games / environments under rewards 

- [ ] Intoducing integration with other custom pygame environments

- [ ] Support for Open AI gym environments 

- [ ] More RL algorithms (other than DQN)

- [ ] Better experimentation managing 

- [ ] Support for stable baseline 
