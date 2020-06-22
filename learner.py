from env import *
import numpy as np
from tqdm import tqdm

class QLearner:
    def __init__(self, env, gamma=.9, alpha=.1):
        self.gamma, self.alpha = gamma, alpha
        self.n_states, self.n_actions = env.n_states, env.actions
        self.Q = np.zeros((self.n_states, self.n_actions))

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

        if policy is None:
            policy = epsilon_greedy

        if winners is not None:
            for player in env.players:
                winners[player] = 0

        for match in tqdm(range(matches)):
            done, s = False, env.reset()

            while not done:
                actions = [
                    policy(Q=self.Q[s], n_actions=self.n_actions, epsilon=epsilon), 
                    policy(Q=self.Q[s], n_actions=self.n_actions, epsilon=epsilon)
                ]
                rews, s_, actions, done, _ = env.step(actions)
                delta = rews[0] + self.gamma * self.Q[s_, np.argmax(self.Q[s_])]
                self.Q[s, actions[0]] = (1 - self.alpha) * self.Q[s, actions[0]] + self.alpha * delta
                s = s_

            epsilon = decay(epsilon)

            if winners is not None:
                winners[env.winner] += 1

            if render and match % interval == 0:
                env.render()

        return self.Q, winners

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
    players = ['P1', 'P2']
    env = ConnectFour(players)
    matches = 25000

    Q, winners = QLearner(env).learn(matches=matches, winners=True)
    
    print('Winners: {}'.format(winners))
