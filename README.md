## Curve Fever - Curve fever

Final project in introduction to artificial inteligernce course at Hebrew university in Jerusalem.

Authors - Aviad Sar-Shalom, Daniel Rotem, Gabrielle Marmur, Mira Finkelstien

Contents
--------

* [Description](#description)
* [Instalation](#installation)
* [Usage](#usage)
* [Support](#support)

Description
-----------

Curve Fever is a popular computer game for 2-4 players. 

**Game flow**
"Each player spawns as a dot at a random spot on the playing field, move at a constant speed. Each player has the ability to turn left or right, although the turning speed is limited such that sharp turns are not possible. As the dot travels across the playing field, it draws a permanent, solid line in its wake, in the color of that player. When the dot collides with any section of line or the boundary of the playing field, the player instantly loses, although the line remains in the playing field until the end of the game. The game becomes increasingly difficult as more of the playing field is blocked off by lines. Other players may try to draw barriers to block the path of other players, forcing them into a collision. However, as the lines are being drawn, gaps are occasionally being generated that can be used to escape a section of the map that has been blocked off. The game is won when all but one of the players has collided." - Wikepadia

<img src="static/img/CurveFever.gif" width="550" height="400"/>


We wrote this project based on an open source python implementation of Curve fever using the pygame library. The original implementation can be found in this [repository](link to repository) 

We designed and trained 2 computer agents that can be initialized as players in the game.
- **Min-Max agent:**

    Preforms a search of the game tree using alpha-beta pruning. It uses a weighted sum of a few heuristis evaluate the nodes of the tree and chooses the best move in each timestep. 
    
- **Deep reinforcement learning Agent:**

    Extracts features from the game state at every time step and evalueates the feature vector using a trained neural net to choose an action.




Be sure to try out the game and test yourself against our DRL agent.

In order to run the game, install all libraries listed in requierments.txt and run main.py with python >=3.7

Feel free to send any comments or questions regarding this project to danielrotem@mail.huji.ac.il

Thank you
In order to run the game, install all libraries specified in requierments.txt and than run main.py with python >= 3.7
Double DQN
=====


description
-----------
A double deep Q-learning library implemented in python3 using [tensorflow](https://www.tensorflow.org/)

Q-learning is is a model-free [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) algorithm 
to learn a policy telling an agent what action to take under what circumstances. 
This library provides the basic structure to learn a policy in an environment using the 
[Double-deep Q-learning](https://arxiv.org/pdf/1509.06461.pdf) algorithm.
 
The code files in this project include: 
- **dqn_agent.py:**
- **double_dqn.py:** 
- **experience_replay.py:** 
- **dqn_env.py:** 
 
installation
--------

### clone
Clone this repository to your local machine using 'repository address goes here'
            
    git clone https://github.com/dayMan33/double_DQN.git

### setup 
while in the project directory, run setup.sh to install all requirements.

    double_dqn> setup.sh

usage
-----
To start training an agent, you must implement a class of dqn_env with the required methods. Only then can you 
initialize a dqn_agent with an instance of the environment as its only argument. Once you have done that, you will need
to set the model of the agent to be a compiled tf.keras Model tailored specifically to your environment's needs. 
After setting the agent's model, you can train it by calling dqn_agent.train with the necessary arguments
 
```python
    from double_dqn.dqn_env import DQNenv
    from double_dqn.dqn_agent import DQNagent

    path = 'model_path'
    num_episodes = N
    env = MyEnv() # Inherits from DQNenv
    agent = DQNagent(env)
    
    # A compiled tf.keral Model to use as the agent's NN.
    model = build_model(env.get_state_shape(), env.get_action_shape())
    agent.set_model(model)
    agent.train(num_episodes, path)
```
    
The train method saves the weights and the model architecture to the specified path

For a more detailed example, check out this [repository](https://github.com/dayMan33/double_dqn_usage.git)

support
-------
For any questions or comments, feel free to email me at danielrotem33@gmail.com


