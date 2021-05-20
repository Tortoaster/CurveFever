import math
import numpy as np
from static.settings import *
from modules.players.player import Player
import neat

MAX_RAY_LENGTH = math.sqrt(ARENA_WIDTH ** 2 + ARENA_HEIGHT ** 2)
SPACING = PLAYER_RADIUS * 2
RAYS = 5
SPREAD = 15

class NeatPlayer(Player):
    def __init__(self, player_id, game, genome, net):
        super().__init__(player_id, game)

        self.genome = genome
        genome.fitness = 0
        self._net = net
        self.predictions = 0
        self.total_time = 0

    def get_action(self, state):
        # The distances of the rays
        inputs = [self.cast_ray((angle - RAYS // 2) * SPREAD) / MAX_RAY_LENGTH for angle in range(RAYS)]
        # Increase fitness each time this function is called
        self.genome.fitness += 1
        # Get the ouputs from the network
        outputs = self.net.activate(inputs)
        if outputs[0] > 0.5 or outputs[1] > 0.5:
            if outputs[0] > outputs[1]:
                return LEFT
            else:
                return RIGHT
        return STRAIGHT
        
    def set_game(self, game):
        self.game = game

    def cast_ray(self, angle):
        position = self.game.state.get_position(self.id)
        angle = self.game.state.get_angle(self.id) + angle

        index = 1
        while True:
            index += 1
            hx = np.cos(angle) * SPACING * index
            hy = np.sin(angle) * SPACING * index
            pos = [hx + position[0], - hy + position[1]]

            if self.game.detect_collision(self.id, self.game.state, pos):
                return index

            # pygame.draw.circle(self.game.window, WHITE, self.game.adjust_pos_to_screen(pos), 1)