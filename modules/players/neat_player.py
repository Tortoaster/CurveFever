import math
import pickle

import numpy as np

from modules.players.player import Player
from static.settings import *

MAX_RAY_LENGTH = 150
SPACING = PLAYER_RADIUS * 2
RAYS = 20
SPREAD = 18


class NeatPlayer(Player):
    def __init__(self, player_id, game, genome=None, net=None):
        super().__init__(player_id, game)

        self.training = game.training_mode
        self.genome = genome
        if genome:
            genome.fitness = 0
        self.net = net or pickle.load(open("static/pickles/neat-656.pickle", 'rb'))
        self.predictions = 0
        self.total_time = 0
        self.record = 0

    def get_action(self, state):
        # The distances of the rays
        # inputs = [self.cast_ray((angle - RAYS // 2) * SPREAD) / MAX_RAY_LENGTH for angle in range(RAYS)]
        angles = [-130, -111, -93, -76, -60, -55, -41, -28, -16, -5, 5, 16, 28, 41, 55, 60, 76, 93, 111, 130]
        inputs = [self.cast_ray(angles[index]) / MAX_RAY_LENGTH for index in range(RAYS)]
        # Get the outputs from the network
        # direction = self.net.activate(inputs)[0]
        outputs_left = self.net.activate(inputs[RAYS // 2:])[0]
        outputs_right = -self.net.activate(list(reversed(inputs[:RAYS // 2])))[0]
        direction = outputs_right + outputs_left

        if self.training:
            # Increase fitness each time this function is called
            self.genome.fitness += 1

        if direction < -0.1:
            return LEFT
        elif direction > 0.1:
            return RIGHT
        return STRAIGHT

    def cast_ray(self, angle):
        position = self.game.state.get_position(self.id)
        angle = self.game.state.get_angle(self.id) + (angle / 180 * math.pi)

        for distance in range(MAX_RAY_LENGTH):
            hx = np.cos(angle) * SPACING * distance
            hy = np.sin(angle) * SPACING * distance
            pos = [hx + position[0], - hy + position[1]]

            if self.game.detect_collision_pos(pos):
                return distance

            # pygame.draw.circle(self.game.window, WHITE, self.game.adjust_pos_to_screen(pos), 1)
