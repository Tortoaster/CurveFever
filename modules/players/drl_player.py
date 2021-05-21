import json

import numpy as np
from tensorflow.keras.models import model_from_json

from modules.players.player import Player


class DRLPlayer(Player):
    def __init__(self, player_id, game, model_path, weight_path):
        super().__init__(player_id, game)
        with open(model_path, 'r') as json_file:
            config = json.load(json_file)
            model = model_from_json(config)
            model.load_weights(weight_path).expect_partial()
        self._net = model
        self.predictions = 0
        self.total_time = 0
        self.alive = True
        

    def get_action(self, state):
        # if self.predictions > 0 and not self.predictions % 100:
        #     print(f'average action takes {self.total_time / self.predictions} seconds')
        self.predictions += 1
        drl_state = state.adjust_to_drl_player(self.id)  # self.crop_box(state.board, state.positions)
        values = self._net(drl_state[np.newaxis, ...], training=False)
        return np.random.choice(np.flatnonzero(values == np.max(values)))
