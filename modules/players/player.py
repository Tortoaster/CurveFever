import abc


class Player(object):
    speed = 4
    radius = 4
    d_theta = 0.09
    no_draw_time = 10

    def __init__(self, player_id, game):
        self.id = player_id
        self.game = game
        self.position = None
        self.angle = None
        self.alive = True
        self.count = 0
        self.color = None

    @abc.abstractmethod
    def get_action(self, state):
        raise NotImplementedError('Please implement this method')
