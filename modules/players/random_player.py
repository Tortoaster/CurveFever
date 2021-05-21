import numpy as np

from modules.players.player import Player


class RandomPlayer(Player):
    def __init__(self, player_id, game):
        super().__init__(player_id, game)
        self.alive = True

    def get_action(self, state):
        return np.random.randint(3)
