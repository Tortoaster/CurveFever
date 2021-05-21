from modules.environment.curve_fever_game import CurveFever
from modules.players.player_factory import PlayerFactory
from modules.players.neat_player import NeatPlayer
import threading
import neat
import pickle
import os

fitnessLock = threading.Lock()
highest_fitness = 0


def eval_genomes(genomes, config):
    numGen = 2
    threads = [genome(genomes[i:i + numGen], i, i + numGen, config) for i in range(0, len(genomes), numGen)]
    for t in threads:
        t.start()
    [t.join() for t in threads]
    print('\n')


class genome(threading.Thread):
    def __init__(self, genomes, begin, end, config):
        threading.Thread.__init__(self)
        self.genomes = genomes
        self.game = CurveFever(training_mode=True)
        self.config = config
        self.begin = begin
        self.end = end
        self.players = self.create_players()

    def create_players(self):
        player_id = 0
        tmp = []
        for _, genome in self.genomes:
            net = neat.nn.RecurrentNetwork.create(genome, self.config)
            tmp.append(NeatPlayer(player_id, self.game, genome, net))
            player_id += 1
        
        # tmp.append(PlayerFactory.create_player('d', self.begin+1, self.game))
        tmp.append(PlayerFactory.create_player('ab', self.begin+2, self.game))
        return tmp

    def run(self):
        print("Range", self.begin, self.end)
        self.game.initialize(self.players)
        self.game.training_loop()
        player = max(self.players, key=lambda k: k.genome.fitness)
        fitness = player.genome.fitness
        check_highest(fitness, player.net)


def check_highest(fitness, net):
    fitnessLock.acquire()
    global highest_fitness

    if fitness > highest_fitness:
        highest_fitness = fitness
        
        print("Highest fitness:", highest_fitness)
        if fitness > 200:
            pickle.dump(net, open(("static/picklesC/neat-" + str(fitness) +  ".pickle"), "wb"))
    fitnessLock.release()


def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'static/picklesC/config.txt')
    run(config_path)
