from env import *
from learner import QLearner
import os, re
import numpy as np
from tqdm import tqdm

def best_player(winners, verbose=False):
    """
    Determines the best player in a dict/map of winners
    @param winners dict/map of winners
    @param verbose True if 
    """
    wc = winners.copy()

    if 'None' in wc:
        del wc['None']

    best_player, best_value = None, 0

    for key, value in wc.items():
        if best_player is None:
            best_player = key
            best_value = value
        
        if value > best_value:
            best_player = key
            best_value = value

    if verbose:
        print('Best player: {}'.format(best_player))

    return best_player

def training_data():
    """
    Traverses training_data directory for the most mature data set,
    """
    data_sets, training_data = [], [f for f in os.listdir('training_data/') if os.path.isfile(os.path.join('training_data/', f))]
    for f in training_data:
        data_set = int(re.sub(r'\..*','', f))
        data_sets.append(data_set)

    return data_sets

def train(env, base, diff):
    print('Best player: {}'.format(best_player(winners)))

def play(env, matches=None):
    top_data = max(training_data())
    data = np.loadtxt('training_data/{}.csv'.format(top_data), delimiter=',')
    winners = QLearner(env, Q=data).play()
    print('Winners: {}'.format(winners))

def run(env, matches):
    data = training_data()
    if matches in data:
        # Play
        play(env, matches)

    elif matches < max(data):
        # Play with best training data
        play(env)
    else:
        # Train then play
        diff = matches - max(data)
        train(env, base, diff)
        play(env)

if __name__ == "__main__":
    np.random.seed(1337) # Seeding data for consistent results

    players = ['Gio', 'P2']
    env = ConnectFour(players)
    matches = 50000

    run(env, matches)
