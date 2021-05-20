from modules.environment.curve_fever_game import CurveFever
from modules.players.neat_player import NeatPlayer
import neat

def eval_genomes(genomes, config):
    neat_players = []
    ## without gui
    # game = CurveFever(training_mode=True)
    ## gui
    game = CurveFever()
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        neat_players.append(NeatPlayer(genome_id, game, genome, net))
    
    # without gui
    # game.set_players(neat_players)
    game.neat_initialize(neat_players)
    game.training_loop()

    ## gui
    game.intro()
    game.neat_initialize(neat_players)
    game.loop()

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
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    run


