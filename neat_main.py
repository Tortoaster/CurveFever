from modules.environment.curve_fever_game import CurveFever
from modules.players.neat_player import NeatPlayer

from threading import Lock
import neat
import pickle
import os

fitnessLock = Lock()
highest_fitness = 0


def eval_genomes(genomes, config):
    global highest_fitness
    neat_players = []
    for i in range(0, len(genomes), 4):
        # for genome_id, genome in genomes:
        game = CurveFever(training_mode=True)
        player_id = 0
        tmp_players = []
        for genome_id, genome in genomes[i:i + 4]:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            player = NeatPlayer(player_id, game, genome, net)
            neat_players.append(player)
            tmp_players.append(player)
            player_id = player_id + 1
        print("Range", i, i + 4)
        game.play(tmp_players)

        player = max(neat_players, key=lambda k: k.genome.fitness)
        fitness = player.genome.fitness
        print(fitness, highest_fitness)

        check_highest(fitness, player.net)


def check_highest(fitness, net):
    fitnessLock.acquire()
    global highest_fitness

    if fitness > highest_fitness:
        highest_fitness = fitness
        # Save network if score improves
        if fitness > 750:
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
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'static/models/config.txt')
    run(config_path)
