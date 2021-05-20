from modules.players.player import Player
import neat

class NeatPlayer(Player):
    def __init__(self, player_id, game, genome, net):
        super().__init__(player_id, game)

        self.genome = genome
        genome.fitness = 0
        self._net = net
        self.predictions = 0
        self.total_time = 0

    def set_game(self, game):
        self.game = game

    def get_action(self, state):
        self.genome.fitness += 1
        return 2
