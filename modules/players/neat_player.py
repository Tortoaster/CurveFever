import math
import pickle

import numpy as np

from modules.players.player import Player
from static.settings import *

MAX_RAY_LENGTH = 30
SPACING = PLAYER_RADIUS * 1.5
RAYS = 21
SPREAD = 18
MAX_DISTANCE = MAX_RAY_LENGTH * SPACING

a = True

class NeatPlayer(Player):
    def __init__(self, player_id, game, genome=None, net=None):
        super().__init__(player_id, game)

        self.training = game.training_mode
        self.genome = genome
        if genome:
            genome.fitness = 0
        self.net = net or pickle.load(open("static/picklesAttackC/neat-2-694.985705807098.pickle", 'rb'))
        self.predictions = 0
        self.total_time = 0
        self.record = 0
        

    def get_action(self, state):
        global a
        # The distances of the rays
        # inputs = [self.cast_ray((angle - RAYS // 2) * SPREAD) / MAX_RAY_LENGTH for angle in range(RAYS)]
        angles = [-145, -126, -108, -91, -75, -60, -46, -33, -21, -10, 0, 10, 21, 33, 46, 60, 75, 91, 108, 126, 145]
        inputs = [self.cast_ray(angles[index]) / MAX_RAY_LENGTH for index in range(RAYS)]
        # Get nearest player (if it exists) and add the relative angle and distance to the inputs
        player_distances = sorted([(self.game.distance_between_two_pos(state.get_position(self.id), state.get_position(p.id)), p) for p in self.game.players if p.id != self.id ], key=lambda x: x[0])
        nearest_player = list(filter(lambda p: state.alive[p[1].id], player_distances))
        if not nearest_player:
            angle_dis_inputs = [0, MAX_DISTANCE, 0]
        else:
            distance = nearest_player[0][0]
            nearest_player = nearest_player[0][1]

            close_enemies = self.check_surroundings(state)

            angle_dis_inputs = [(state.get_angle(self.id) % (2 * np.pi)) - (state.get_angle(nearest_player.id) % (2 * np.pi)) - np.pi, min(distance, MAX_RAY_LENGTH), close_enemies]
        if a:
            a = False
            print("players :", player_distances, "nearest players :", nearest_player, "ANGLE DIS:", angle_dis_inputs)

        # Get the outputs from the network
        # direction = self.net.activate(inputs)[0]
        outputs_left = self.net.activate(list(reversed(inputs[:RAYS // 2 + 1])) + angle_dis_inputs)[0]
        angle_dis_inputs[0] = -angle_dis_inputs[0]
        outputs_right = -self.net.activate(inputs[RAYS // 2:] + angle_dis_inputs)[0]
        direction = outputs_right + outputs_left

        # Give player extra fitness when in proximity of other players
        if self.training:
            # Increase fitness each time this function is called
            self.genome.fitness += 1 + (self.check_surroundings(state))

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

    def check_surroundings(self, state):
        angle = state.get_angle(self.id) + (1/2 * math.pi)
        fitness_adjust = 0

        slope = np.tan(angle)

        pos = state.get_position(self.id)
        b = pos[1] - (slope * pos[0])

        players = self.game.players
        for player in players:
            if not player.id == self.id:
                pos2 = state.get_position(player.id)
                distance = self.game.distance_between_two_pos(pos, pos2)
                if distance < MAX_DISTANCE:
                    x = pos2[0]
                    y = pos2[1]
                    if slope >= 0:
                        if y < (slope * x + b):
                            fitness_adjust -= ((MAX_DISTANCE - distance) / MAX_DISTANCE) ** 2 * 10
                        else:
                            fitness_adjust += ((MAX_DISTANCE - distance) / MAX_DISTANCE) ** 2 * 5
                    if slope < 0:
                        if y > (slope * x + b):
                            fitness_adjust -= ((MAX_DISTANCE - distance) / MAX_DISTANCE) ** 2 * 10
                        else:
                            fitness_adjust += ((MAX_DISTANCE - distance) / MAX_DISTANCE) ** 2 * 5
        return fitness_adjust
