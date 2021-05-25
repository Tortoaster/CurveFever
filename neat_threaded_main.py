from modules.environment.curve_fever_game import CurveFever
from modules.players.neat_player import NeatPlayer
import threading
import neat
import pickle
import os

fitnessLock = threading.Lock()
highest_fitness = 0
PLAYERS = 4


def eval_genomes(genomes, config):
    threads = [Genome(genomes[i:i + PLAYERS], i, i + PLAYERS, config) for i in range(0, len(genomes), PLAYERS)]
    for t in threads:
        t.start()


class Genome(threading.Thread):
    def __init__(self, genomes, begin, end, config):
        threading.Thread.__init__(self)
        self.genomes = genomes
        self.game = CurveFever(training_mode=True)
        self.config = config
        self.players = self.create_players()
        self.begin = begin
        self.end = end

    def create_players(self):
        player_id = 0
        tmp = []
        for _, genome in self.genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            tmp.append(NeatPlayer(player_id, self.game, genome, net))
            player_id += 1
        return tmp

    def run(self):
        print("Range", self.begin, self.end)
        self.game.initialize(self.players)
        self.game.training_loop()
        player = max(self.players, key=lambda k: k.Genome.fitness)
        fitness = player.genome.fitness
        check_highest(fitness, player.net)


def check_highest(fitness, net):
    fitnessLock.acquire()
    global highest_fitness

    if fitness > highest_fitness:
        highest_fitness = fitness
        
        print("Highest fitness:", highest_fitness)
        pickle.dump(net, open(("static/pickles/neat-" + str(fitness) + ".pickle"), "wb"))
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
    print('\nBest Genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'static/models/config.txt')
    run(config_path)
