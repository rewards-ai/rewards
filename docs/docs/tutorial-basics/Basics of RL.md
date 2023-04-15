AI, or artificial intelligence, refers to the development of computer systems that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

**Reinforcement learning (RL)** is a type of machine learning that enables an agent to learn through trial and error by interacting with an environment. In RL, an agent learns to take actions that maximize a reward signal, by exploring the environment and receiving feedback in the form of positive or negative rewards.

While reinforcement learning is a subfield of AI, it differs from other machine learning techniques such as supervised and unsupervised learning. In **supervised learning**, a model is trained using labeled data, while in **unsupervised learning**, the model identifies patterns and relationships in the data without any explicit guidance. **Reinforcement learning**, on the other hand, involves an agent learning to take actions based on feedback from the environment, rather than pre-labeled or unstructured data.


## Terminologies

Now before getting started with installation and your hand's on the `rewards sdk`, lets get to know about some basic terminologies that are frequently used while learning Reinforcement learning 
and Machine learning. These terminologies are essential to understand reinforcement learning, and will form the foundation of your understanding as you dive deeper into the field. By grasping these concepts, you'll be well on your way to building your own RL agents and exploring the exciting possibilities of this dynamic field. So let's get started. 

### Agent
An agent is the entity that learns and makes decisions based on its interaction with the environment. It takes in information about the state of the environment and outputs actions to take. 

For example, in a game of chess ♟️, the player controlling the pieces is the agent. In a game of car race 🏎️, the person controlling the car is the agent. Similarly other than games or sports, suppose you want to given a task to clean the house. Then you are the agent here. In reinforcement learning, this agent is **NOT** Human, rather the machine itself. Through some sets of algorithms, we make the computer mimic the behaviour logically similarly like how a real human does. Sounds interesting right? 


### Environment
The environment is the context in which the agent operates. It consists of all the elements that the agent can perceive, and may include things like game boards, obstacles, or other players. 

For example, the chess board and pieces in a game of chess form the environment. Similarly, in a car race game, environment becomes my race track. And similarly, in the previous task where an agnet
was assigned to clean the house, there the environment is the house itself.


### Actions
In reinforcement learning, an agent interacts with an environment by taking actions. Actions are the decisions or choices made by the agent in response to a given state of the environment.

For example, in a game of chess, the agent's actions would be the different moves it can make based on the current board state. In a self-driving car, the agent's actions would be the decisions it makes on when to accelerate, brake, or turn based on its sensors and the surrounding environment. Similarly in the task to clean the house, the agent's action will be to move here and there in the house (without breaking any stuff) to clean the areas

The goal of reinforcement learning is for the agent to learn the optimal actions to take in a given environment in order to maximize its rewards. By taking actions and receiving feedback in the form of rewards, the agent gradually learns which actions lead to better outcomes, and improves its decision-making over time.


### Reward
A reward is a signal that indicates how well the agent is performing. It's the feedback that the agent receives from the environment in response to its actions. The goal of the agent is to maximize the total reward it receives over time. For example, in a game of chess, the reward could be the number of pieces captured by the player.

### Reward function
A reward function is a function that maps each possible state and action pair to a reward value. It's used to evaluate the performance of the agent and guide its learning. For example, a reward function for a chess-playing agent might assign a positive reward for capturing an opponent's piece and a negative reward for losing its own.

Lets take another example of the reward function. Suppose we are creating a self driving car agent. The main task of the car is to follow certain path and follow along. Now how we are gonna
give the feedback to the car? You have to create some sets of rules to penalize the agent and when you are gonna reward the agent. Some situations where:

- The agent goes to different directions 
- Crashes with some objects
- Do not reaches in time

Are some of the situations where you are gonna penalize the agent. At the same time, there are some situations like:

- The agent reaches in time
- Agent follows the path 
- Agent does not crashes and rides safes.

These are the situations where the agent is gonna rewarded with something. Now these rewards will be some mathematical values. Better the job better the reward given. More number of penalties will given for the vice versa. 

In a typical reward function, if you have to see what is the agent's current observation in the environment. For the game of chess it will be what is the configuration of the chess board. In the case of self driving car, it will be what is the current observation of the car position and it's surrounding environment. Now based on observations you will focus on creating some checks and will that will generate the corresponding reward. 

So let's take an example of a car game. In one time instant the car has it following observation. 

- The car is alive (not crashed)
- The car has some co-ordinates of it's center of mass
- The car also measures the distance between it's center of mass and all the sides of road

Now based on those observations, as all these observation you will provide a positive reward if it passes each checks and it will provide a negative reward for each negative checks. 