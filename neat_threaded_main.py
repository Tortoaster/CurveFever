from modules.environment.curve_fever_game import CurveFever
from modules.players.neat_player import NeatPlayer
import matplotlib.pyplot as plt
import threading
import neat
import pickle
import os
import random

fitnessLock = threading.Lock()
highest_fitness = 0
PLAYERS = 4
current_generation = 0
record_generations = []
record_fitnesses = []

output_folder = "static/pickles/"

def eval_genomes_tournament(genomes, config):
    global current_generation

    play(genomes, config, 2)
    in_tournament = genomes
    i = 0
    while(len(in_tournament)>PLAYERS):
        print("Round " + str(i) + "\n")
        in_tournament = sorted(in_tournament, key=lambda k: k[1].fitness, reverse=True)[:len(in_tournament)//PLAYERS]
        for _, g in in_tournament:
            g.fitness = 0
        play(in_tournament, config, 2)
        i += 1

    current_generation += 1

def eval_genomes_shuffle_3_games(genomes, config):
    global current_generation

    games = 3
    
    threads = []
    for i in range(0,games):
        random.shuffle(genomes)
        threads += [genome(genomes[i:i + PLAYERS], i, i + PLAYERS, config) for i in range(0, len(genomes), PLAYERS)]

    print("Starting", len(threads), "number of matches with", games, " games")

    for t in threads:
        t.start()
    [t.join() for t in threads]

    for _,g in genomes:
        g.fitness /= games

    winner = max(genomes, key = lambda k: k[1].fitness)
    check_highest(winner, config)
    current_generation += 1

def play(in_tournament, config, games):
    threads = []
    for i in range(0,games):
        threads = [genome(in_tournament[i:i + PLAYERS], i, i + PLAYERS, config) for i in range(0, len(in_tournament), PLAYERS)]
    # threads += [genome(in_tournament[i:i + PLAYERS], i, i + PLAYERS, config) for i in range(0, len(in_tournament), PLAYERS)]
    for t in threads:
        t.start()
    [t.join() for t in threads]

    for _,g in in_tournament:
        g.fitness /= games

    winner = max(in_tournament, key = lambda k: k[1].fitness)
    
    check_highest(winner, config)

def save_plot(stats):
    # plt.plot(record_generations, record_fitnesses)
    plt.plot(stats.get_fitness_stat(max))
    plt.plot(stats.get_fitness_mean())

    plt.xlabel("Generations")
    plt.ylabel("Fitness")

    # plt.legend(['global best', 'generation best', 'generation mean'], loc='upper left')
    plt.legend(['generation best', 'generation mean'], loc='upper left')
    plt.savefig(output_folder + "plot.png")

class genome(threading.Thread):
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
        # print("Range", self.begin, self.end)
        self.game.initialize(self.players)
        self.game.training_loop()
        # player = max(self.players, key=lambda k: k.genome.fitness)


def check_highest(winner, config):
    fitnessLock.acquire()
    global highest_fitness

    fitness = winner[1].fitness
    record_generations.append(current_generation)
    record_fitnesses.append(max(highest_fitness, fitness))

    if fitness > highest_fitness:
        highest_fitness = fitness

        print("Highest fitness:", highest_fitness)
    if fitness >= 250:
        net = neat.nn.FeedForwardNetwork.create(winner[1], config)
        pickle.dump(net, open(( output_folder + "neat-" + str(current_generation) + "-" + str(fitness) + ".pickle"), "wb"))
    fitnessLock.release()


def run(config_file):
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
    
    # Run
    winner = p.run(eval_genomes_shuffle_3_games, 25)

    save_plot(stats)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'static/models/config.txt')
    run(config_path)
