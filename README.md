## Curve Fever - Achtung Die!

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

**Game flow:**

"Each player spawns as a dot at a random spot on the playing field, move at a constant speed. Each player has the
ability to turn left or right, although the turning speed is limited such that sharp turns are not possible. As the dot
travels across the playing field, it draws a permanent, solid line in its wake, in the color of that player. When the
dot collides with any section of line or the boundary of the playing field, the player instantly loses, although the
line remains in the playing field until the end of the game. The game becomes increasingly difficult as more of the
playing field is blocked off by lines. Other players may try to draw barriers to block the path of other players,
forcing them into a collision. However, as the lines are being drawn, gaps are occasionally being generated that can be
used to escape a section of the map that has been blocked off. The game is won when all but one of the players has
collided." - Wikepadia

<img src="static/img/CurveFever.gif" width="550" height="400"/>


We wrote this project based on an open source python implementation of Curve fever using the pygame library. The
original implementation can be found in this [repository](https://github.com/Valaraucoo/AchtungDiePython.git)

We designed and trained 2 computer agents that can be initialized as players in the game.

- **Min-Max agent:**

  Preforms a search of the game tree using alpha-beta pruning. It uses a weighted sum of a few heuristis evaluate the
  nodes of the tree and chooses the best move in each timestep.

- **Deep reinforcement learning Agent:**

  Extracts features from the game state at every time step and evalueates the feature vector using a trained neural net
  to choose an action.

installation
--------

### clone

Clone this repository to your local machine using 'https://github.com/dayMan33/CurveFever.git'

    git clone https://github.com/dayMan33/CurveFever.git

### setup

while in the project directory, run setup.sh to install all requirements.

    CurveFever> setup.sh

Once all dependencies are satisfied, You can run main.py using pytohn>=3.7

usage
-----
Choose at least 2 players and start playing. Try yourself against both agents and see how you compare

support
-------
For any questions or comments, feel free to email me at danielrotem33@gmail.com


