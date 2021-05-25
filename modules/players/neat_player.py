import math
import pickle

import numpy as np

from modules.players.player import Player
from static.settings import *

MAX_RAY_LENGTH = 30
SPACING = PLAYER_RADIUS * 1.5
RAYS = 21
SPREAD = 18


class NeatPlayer(Player):
    def __init__(self, player_id, game, genome=None, net=None):
        super().__init__(player_id, game)

        self.training = game.training_mode
        self.genome = genome
        if genome:
            genome.fitness = 0
        self.net = net or pickle.load(open("static/pickles/neat-870.pickle", 'rb'))
        self.predictions = 0
        self.total_time = 0
        self.record = 0

    def get_action(self, state):
        # The distances of the rays
        # inputs = [self.cast_ray((angle - RAYS // 2) * SPREAD) / MAX_RAY_LENGTH for angle in range(RAYS)]
        angles = [-145, -126, -108, -91, -75, -60, -46, -33, -21, -10, 0, 10, 21, 33, 46, 60, 75, 91, 108, 126, 145]
        inputs = [self.cast_ray(angles[index]) / MAX_RAY_LENGTH for index in range(RAYS)]
        # Get the outputs from the network
        # direction = self.net.activate(inputs)[0]
        outputs_left = self.net.activate(list(reversed(inputs[:RAYS // 2 + 1])))[0]
        outputs_right = -self.net.activate(inputs[RAYS // 2:])[0]
        direction = outputs_right + outputs_left

        if self.training:
            # Increase fitness each time this function is called
            self.genome.fitness += 1

        if direction < 0:
            return LEFT
        else:
            return RIGHT

    def cast_ray(self, angle):
        position = self.game.state.get_position(self.id)
        angle = self.game.state.get_angle(self.id) + (angle / 180 * math.pi)

        for distance in range(1, MAX_RAY_LENGTH):
            hx = np.cos(angle) * SPACING * distance
            hy = np.sin(angle) * SPACING * distance
            pos = [hx + position[0], - hy + position[1]]

            if self.game.detect_collision_pos(pos):
                return distance

            # if not self.training:
            #     pygame.draw.circle(self.game.window, WHITE, self.game.adjust_pos_to_screen(pos), 1)
        return MAX_RAY_LENGTH
