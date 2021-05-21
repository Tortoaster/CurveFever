import abc

import numpy as np

from static.settings import *


class State:
    count = 0

    def __init__(self, players, shape, positions, angles, colors):
        self.players = players
        for i, player in enumerate(players):
            player.position = positions[i]
            player.angle = angles[i]
            player.color = colors[i]
        self.margin = 6
        shape_3d = shape[0] + (self.margin * 2), shape[1] + (self.margin * 2), shape[2]
        self._rgb_board = np.ones(shape_3d) * WHITE
        self.reset_arena()

    def get_3d_pixel(self, coord):
        return self._rgb_board[self.margin + int(round(coord[1])), self.margin + int(round(coord[0])), ...]

    def _get_rgb_board(self):
        return self._rgb_board

    # def set_board(self, board: np.ndarray):
    #     arena = self.get_rgb_board()
    #     arena[...] = board.copy()

    def get_position(self, player_id):
        return self.players[player_id].position

    def get_all_positions(self):
        return [player.position for player in self.players]

    def get_all_angles(self):
        return [player.angle for player in self.players]

    def set_position(self, player_id, position):
        self.players[player_id].position = position

    def adjust_pos_to_board_with_margin(self, position):
        return position[0] + self.margin, position[1] + self.margin

    def get_angle(self, player_id):
        return self.players[player_id].angle

    def set_angle(self, player_id, angle):
        self.players[player_id].angle = angle

    def reset_arena(self):
        self._rgb_board[self.margin:self.margin + ARENA_HEIGHT, self.margin:self.margin + ARENA_WIDTH, ...] = BLACK

    def adjust_to_drl_player_no_position(self, player_id):
        features = self.adjust_to_drl_player(player_id)
        return features[:-2]

    def adjust_to_drl_player(self, player_id):
        # Feature representation
        max_dist = 350
        num_angles = 25
        return self.get_distance_to_obstacles(self.get_angle(player_id), self.get_position(player_id), num_angles, max_dist)

    @abc.abstractmethod
    def update_player_graphics(self, player_id):
        raise NotImplementedError('Please implement this method')

    def draw_circle(self, color, center, radius):
        circle = CIRCLES[radius - 1]
        circle = np.array(circle) + np.array(center)
        circle = np.round(circle).astype(np.int)
        circle = self.clip(circle)
        self._rgb_board[circle[..., 1] + self.margin, circle[..., 0] + self.margin, ...] = color

    def clip(self, circle):
        circle[circle < 0] = 0
        circle[circle[..., 0] >= ARENA_WIDTH, 0] = ARENA_WIDTH - 1
        circle[circle[..., 1] >= ARENA_HEIGHT, 1] = ARENA_HEIGHT - 1
        return np.round(circle).astype(np.int)

    def draw_player(self, player_id, use_color=False):
        self.players[player_id].count += 1
        rgb_color = self.players[player_id].color if use_color else BLACK
        try:
            self.draw_circle(rgb_color, self.players[player_id].position, PLAYER_RADIUS)
        except:
            self.draw_circle(rgb_color, self.players[player_id].position, PLAYER_RADIUS)

    def is_terminal_state(self):
        return sum([player.alive for player in self.players]) == 0

    def get_distance_to_obstacles(self, initial_angle, position, num_angles, max_distance=150):
        angles = [(-np.pi / 2) + ((i / (num_angles - 1)) * np.pi) for i in range(num_angles)]
        angles = initial_angle + np.array(angles)
        distances = np.zeros_like(angles)
        for i, angle in enumerate(angles):
            distances[i] = self.distance_to_obstacle(position, angle, max_distance)
        return distances / max_distance

    def distance_to_obstacle(self, position, angle, max_distance=150):
        position = np.array(position) + self.margin
        distances = np.arange(10, max_distance)
        xx = position[0] + (np.cos(angle) * distances)
        yy = position[1] - (np.sin(angle) * distances)
        xx = np.round(xx).astype(np.int)
        yy = np.round(yy).astype(np.int)
        xx[xx < self.margin] = self.margin - 1
        yy[yy < self.margin] = self.margin - 1
        xx[xx >= ARENA_WIDTH + (self.margin * 2)] = ARENA_WIDTH + (self.margin * 2) - 1
        yy[yy >= ARENA_HEIGHT + (self.margin * 2)] = ARENA_HEIGHT + (self.margin * 2) - 1
        nonzero = np.nonzero(self._rgb_board[yy, xx, ...])
        if len(nonzero[0]) > 0:
            return np.linalg.norm(position - np.array([xx[nonzero[0][0]], yy[nonzero[0][0]]]))
        return max_distance

    def get_player_drl_features(self, player_id):
        features = [self.players[player_id].position[0] / ARENA_WIDTH, self.players[player_id].position[1] / ARENA_HEIGHT]
        return features
