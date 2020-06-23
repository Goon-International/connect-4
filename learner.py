from env import *
import os
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

class QLearner:
    def __init__(self, env, gamma=.9, alpha=.1, Q=None):
        self.gamma, self.alpha = gamma, alpha
        self.n_states, self.n_actions = env.n_states, env.actions
        
        if Q is None:
            self.Q = {}
            for player in env.players:
                self.Q[player] = np.zeros((self.n_states, self.n_actions))
        else:
            self.Q = Q

    def learn(self, matches, epsilon=1., winners=None, policy=None, render=False, interval=1,):
        """
        Learns env game over the number of matches passed in. Env designed similar to Gym env.
        @param matches number of episodes
        @param epsilon value for e-greedy alg.
        @param winners map/dict containing number of wins for each player
        @param policy policy that determines actions
        @param render if True displays end game state
        @param interval frequency of rendering
        @return Q table after matches have been played, winners map/dict
        """
        winners = { 'None': 0 }
        p1, p2 = env.players

        if policy is None:
            policy = epsilon_greedy

        if winners is not None:
            for player in env.players:
                winners[player] = 0

        for match in tqdm(range(matches)):
            done, s = False, env.reset()

            while not done:
                actions = [
                    policy(Q=self.Q[p1][s], n_actions=self.n_actions, epsilon=epsilon), 
                    policy(Q=self.Q[p2][s], n_actions=self.n_actions, epsilon=epsilon)
                ]
                rews, s_, actions, done, _ = env.step(actions)
                delta_p1 = rews[0] + self.gamma * self.Q[p1][s_, np.argmax(self.Q[p1][s_])]
                delta_p2 = rews[1] + self.gamma * self.Q[p2][s_, np.argmax(self.Q[p2][s_])]
                self.Q[p1][s, actions[0]] = (1 - self.alpha) * self.Q[p1][s, actions[0]] + self.alpha * delta_p1
                self.Q[p2][s, actions[1]] = (1 - self.alpha) * self.Q[p2][s, actions[1]] + self.alpha * delta_p2
                s = s_

            epsilon = decay(epsilon)

            if winners is not None:
                winners[env.winner] += 1

            if render and match % interval == 0:
                env.render()

        return self.Q, winners

    def play(self, epsilon=.001, policy=None):
        """
        Plays game vs AI where player moves first.
        @param epsilon value for e-greedy alg.
        @param policy policy that determines actions
        @return winner of match
        """
        if policy is None:
            policy = epsilon_greedy

        done, s = False, env.reset()

        env.render()
        while not done:
            actions = [
                int(input('Choose an action: [0-6]:')),
                policy(Q=self.Q[s], n_actions=self.n_actions, epsilon=epsilon)
            ]

            r_, s_, actions, done, _ = env.step(actions) 
            s = s_

            env.render()

        return env.winner

class DoubleQLearner:
    def __init__(self, env, gamma=.9, alpha=.1):
        self.gamma, self.alpha = gamma, alpha
        self.n_states, self.n_actions = env.n_states, env.actions
        self.Q1 = np.zeros((self.n_states, self.n_actions))
        self.Q2 = np.zeros((self.n_states, self.n_actions))
        self.Qf = np.zeros((self.n_states, self.n_actions))

    def learn(self, matches, epsilon=1., winners=None, policy=None, render=False, interval=1,):
        """
        Learns env game over the number of matches passed in. Env designed similar to Gym env.
        @param matches number of episodes
        @param epsilon value for e-greedy alg.
        @param winners map/dict containing number of wins for each player
        @param policy policy that determines actions
        @param render if True displays end game state
        @param interval frequency of rendering
        @return Final Q table after matches have been played, winners map/dict
        """
        winners = { 'None': 0 }

        if policy is None:
            policy = epsilon_greedy

        if winners is not None:
            for player in env.players:
                winners[player] = 0

        for match in tqdm(range(matches)):
            done, s = False, env.reset()

            while not done:
                self.Qf = (self.Q1 + self.Q2) / 2.
                actions = [
                    policy(Q=self.Qf[s], n_actions=self.n_actions, epsilon=epsilon), 
                    policy(Q=self.Qf[s], n_actions=self.n_actions, epsilon=epsilon)
                ]
                rews, s_, actions, done, _ = env.step(actions)

                if np.random.rand() < .5:
                    delta = rews[0] + self.gamma * self.Q1[s_, np.argmax(self.Q2[s_])]
                    self.Q1[s, actions[0]] = (1 - self.alpha) * self.Q1[s, actions[0]] + self.alpha * delta
                else:
                    delta = rews[0] + self.gamma * self.Q2[s_, np.argmax(self.Q1[s_])]
                    self.Q2[s, actions[0]] = (1 - self.alpha) * self.Q2[s, actions[0]] + self.alpha * delta

                s = s_

            epsilon = decay(epsilon)

            if winners is not None:
                winners[env.winner] += 1

            if render and match % interval == 0:
                env.render()

        return self.Qf, winners

if __name__ == "__main__":
    np.random.seed(1337) # Seeding data for consistent results

    players = ['P1', 'P2']
    env = ConnectFour(players)
    matches = 50000

    # if os.path.isfile('training_data/{}.csv'.format(matches)):
    # data = np.loadtxt('training_data/{}.csv'.format(matches), delimiter=',')
    # else:
    data = None

    Q, winners = QLearner(env, Q=data).learn(matches=matches, winners=True)

    # winners = QLearner(env, Q=data).play()

    # if not os.path.isfile('training_data/{}.csv'.format(matches)):
    np.savetxt('training_data/{}.csv'.format(matches), Q[best_player(winners)], delimiter=',')

    print('Winners: {}'.format(winners))
    print('Best player: {}'.format(best_player(winners)))