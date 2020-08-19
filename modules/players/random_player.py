import numpy as np
from modules.players.player import Player


class RandomPlayer(Player):

    def get_action(self, state):
        return np.random.randint(3)
