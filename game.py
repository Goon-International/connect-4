from env import *
from learner import QLearner
import sys, getopt, os, re
import numpy as np

def training_data():
    """
    Traverses training_data directory for the data sets inside
    @return list of data sets
    """
    data_sets, training_data = [], [f for f in os.listdir('training_data/') if os.path.isfile(os.path.join('training_data/', f))]
    for f in training_data:
        data_set = int(re.sub(r'\..*','', f))
        data_sets.append(data_set)

    return data_sets

def train(env, base, diff):
    """
    Trains AI on any missing data, then saves.
    @env game environment
    @base highest number of matches recorded
    @diff number of matches to train AI against
    """
    data = np.loadtxt('training_data/{}.csv'.format(base), delimiter=',') if base > 0 else None
    old_eps = float(open('epsilons/{}.txt'.format(base), 'r').read()) if base > 0 else 1.
    
    print('Existing epsilon for {} matches: {}'.format(base, old_eps))

    Q, winners, eps = QLearner(env, Q=data).learn(diff, epsilon=old_eps)

    print('Current epsilon: {}'.format(eps))
    print('Winners: {}'.format(winners))
    print('Saving training data on {} matches.'.format(base + diff))

    np.savetxt('training_data/{}.csv'.format(base + diff), Q, delimiter=',')
    new_eps = open('epsilons/{}.txt'.format(base + diff), 'w')
    new_eps.write('{}'.format(eps))
    new_eps.close()

def play(env, matches=None):
    """
    Starts match with player vs AI.
    @param env game environment
    @param matches set of training data to play against
    """
    if matches is None:
        matches = max(training_data())
    data = np.loadtxt('training_data/{}.csv'.format(matches), delimiter=',')
    winners = QLearner(env, Q=data).play()
    print('Winner: {}'.format(winners))

def run(matches, play_game=True, verbose=False):
    """
    Runs program and determines whether to train first or play the game immediately.
    @param env game environment
    @param matches number of matches played in training data
    @param verbose if True logs information about training/playing
    """
    env = ConnectFour(['P1', 'P2'])
    data = [0] if not training_data() else training_data()

    if matches in data:
        if play_game:
            if verbose:
                print('Playing match based off of {} matches. Skipping training.'.format(matches))
            
            play(env, matches)
    elif matches < max(data):
        if play_game:
            if verbose:
                print('Couldn\'t find data on {} matches. Instead choosing {} matches. Skipping training.'.format(matches, max(data)))
            
            play(env)
    else:
        diff = matches - max(data)
        
        if verbose:
            print('Starting training on an additional {} matches.'.format(diff))
        
        train(env, max(data), diff)
        
        if play_game:
            if verbose:
                print('Playing match based off of {} matches.'.format(matches))
            
            play(env)

def main(argv):
    """
    Parses input arguments to determine whether target or background is used.
    @param argv command-line arguments
    @return target and background
    """
    matches, play_game = 10000, True

    opts, args = getopt.getopt(argv,"m:p:",["matches=","play_game="])

    for opt, arg in opts:
        if opt in ("-m", "--matches"):
            matches = int(arg)
        if opt in ("-p", "--play_game"):
            play_game = bool(arg)

    if opts == [] and args == []:
        print('No options specified.')
        print('Usage [short]: python3 game.py -m 10000 -p True')
        print('Usage [long]: python3 game.py --matches=20000 --play_game=False\n')

    return matches, play_game

if __name__ == "__main__":
    np.random.seed(1337) # Seeding data for consistent results
    player = 'Bob'

    matches, play_game = main(sys.argv[1:])

    run(matches, play_game=play_game, verbose=True)