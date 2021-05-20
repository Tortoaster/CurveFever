import math
import numpy as np
from static.settings import *
from modules.players.player import Player
import neat
import pickle

MAX_RAY_LENGTH = math.sqrt(ARENA_WIDTH ** 2 + ARENA_HEIGHT ** 2)
SPACING = PLAYER_RADIUS * 2
RAYS = 7
SPREAD = 30

class NeatPlayer(Player):
    def __init__(self, player_id, game, genome = None, net = None):
        super().__init__(player_id, game) 

        self.training = game.training_mode
        self.genome = genome
        if genome:
            genome.fitness = 0
        self.net = net or pickle.load(open("pickles/neat-2488.pickle", 'rb'))
        self.predictions = 0
        self.total_time = 0
        self.record = 0

    def get_action(self, state):
        # The distances of the rays
        inputs = [self.cast_ray((angle - RAYS // 2) * SPREAD) / MAX_RAY_LENGTH for angle in range(RAYS)]
        # Get the ouputs from the network
        outputs = self.net.activate(inputs)

        if self.training:
            # Increase fitness each time this function is called
            self.genome.fitness += 1

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
        angle = self.game.state.get_angle(self.id) + (angle / 180 * math.pi)

        index = 1
        while True:
            index += 1
            hx = np.cos(angle) * SPACING * index
            hy = np.sin(angle) * SPACING * index
            pos = [hx + position[0], - hy + position[1]]

            if self.game.detect_collision(self.id, self.game.state, pos):
                return index

            # pygame.draw.circle(self.game.window, WHITE, self.game.adjust_pos_to_screen(pos), 1)
