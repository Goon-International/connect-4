import numpy as np

def epsilon_greedy(Q, n_actions, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
        amax, n_max = np.argmax(Q), n_actions - 1
        return amax if amax <= n_max else n_max

def decay(eps):
    decay, min_bound = .995, .001
    new_eps = eps * decay
    
    return new_eps if new_eps > min_bound else min_bound

class ConnectFour:
    def __init__(self, players):
        self.columns, self.rows,  = 7, 6
        self.positions = self.init_positions()
        self.n_states = len(self.positions) ** 3
        self.actions, self.players = self.columns, players
        self.board = np.zeros((self.columns, self.rows), dtype=int)
        self.stacks = np.ones((self.columns), dtype=int) * self.rows - 1
        self.winner,  self.done = 'None', False

    def init_positions(self):
        positions, pos_id = {}, 1
        
        for col in range(self.columns):
            for row in range(self.rows):
                positions[(col, row)] = pos_id
                pos_id += 1
        
        return positions

    def get_state(self):
        state_values = np.zeros((self.columns, self.rows))

        for col in range(self.columns):
            for row in range(self.rows):
                state_values[col, row] = self.positions[(col, row)] * self.board[col, row]
        
        state = int(np.sum(state_values))

        if state >= self.n_states:
            print('Warning: State {} is greater than # of states'.format(state))
        
        return state

    def insert(self, position, player, verbose=False):
        if verbose:
            print('Inserting into position {}'.format(position))

        if self.stacks[position] >= 0:
            self.board[position, self.stacks[position]] = self.players.index(player) + 1
            self.stacks[position] -= 1
            return True
        else:
            return False

    def on_board(self, player_pos, coords):
        is_on_board = True

        for coord in coords:
            is_on_board = is_on_board and self.board[coord[0], coord[1]] == player_pos
        
        return is_on_board

    def is_winner(self, player):
        player_pos = self.players.index(player) + 1

        for col in range(3, self.columns):
            for row in range(0, 2):
                right_diag = self.on_board(player_pos, [(col, row), (col-1, row+1), (col-2,row+2), (col-3, row+3)])
        
        for col in range(3, self.columns):
            for row in range(0, 3):
                left_diag = self.on_board(player_pos, [(col, row), (col-1, row-1), (col-2,row-2), (col-3, row-3)])

        for col in range(self.columns):
            for row in range(0, self.rows-3):
                right = self.on_board(player_pos, [(col, row), (col, row+1), (col, row+2), (col, row+3)])

        for col in range(0, self.columns-3):
            for row in range(self.rows):
                up = self.on_board(player_pos, [(col, row), (col+1, row), (col+2, row), (col+3, row)])

        is_winner = right_diag or left_diag or right or up

        if is_winner:
            self.winner = player

        return is_winner
    
    def is_tie(self):
        is_tie = np.amin(self.board.flatten()) != 0

        if is_tie:
            self.winner = 'None'

        return is_tie

    def reset(self):
        self.board = np.zeros((self.columns, self.rows), dtype=int)
        self.stacks = np.ones((self.columns), dtype=int) * self.rows - 1
        self.winner, self.done = None, False

    def step(self, actions, policy=None):
        if policy is None:
            policy = np.random.randint

        rewards = np.ones((len(self.players))) * -.01

        for index, player in enumerate(self.players):
            inserted = False

            while not inserted:
                if self.insert(actions[index], player):
                    inserted = True
                else:
                    actions[index] = policy(self.actions)
            
            if self.is_winner(player):
                complement = (index + 1) % 2
                rewards[index], rewards[complement] = 1., -1.
                self.done = True
                break
            elif self.is_tie():
                rewards *= 0
                self.done = True
                break

        return rewards, self.get_state(), actions, self.done, self.info(rewards)

    def info(self, rewards):
        info = 'Info:'
        info += '\nState: {}'.format(self.get_state())
        info += '\nRewards: {}'.format(rewards)

        return info

    def render(self):
        board, players = '', ['None'] + self.players

        print('~*~ Connect 4 ~*~')
        print('State:', self.get_state())
        
        for row in range(self.rows):
            for col in range(self.columns):
                board += '| {} |'.format(self.board[col, row])
            board += '\n'

        print(board)
        print('Winner: {} ({})\n'.format(self.winner, players.index(self.winner)))
