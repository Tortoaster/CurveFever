import numpy as np

from modules.players.player import Player


class RandomPlayer(Player):
    def __init__(self, player_id, game):
        super().__init__(player_id, game)
        
        self.alive = True
        self.action = STRAIGHT
        self.position = None
        self.angle = None
        self.count = 0
        self.color = None

    def get_action(self, state):
        return np.random.randint(3)
