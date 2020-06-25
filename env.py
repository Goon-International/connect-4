import numpy as np

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
        """
        Initializes all board positions.
        @return dict/map of positions
        """
        positions, pos_id = {}, 1
        
        for col in range(self.columns):
            for row in range(self.rows):
                positions[(col, row)] = pos_id
                pos_id += 1
        
        return positions

    def get_state(self):
        """
        Gets discrete state representation of game board.
        @return state
        """
        state_values = np.zeros((self.columns, self.rows))

        for col in range(self.columns):
            for row in range(self.rows):
                state_values[col, row] = self.positions[(col, row)] * self.board[col, row]
        
        state = int(np.sum(state_values))

        if state >= self.n_states:
            print('Warning: State {} is greater than # of states'.format(state))
        
        return state

    def insert(self, position, player, verbose=False):
        """
        Inserts a piece into the board for said player.
        @param position the coordinate where a piece is to be placed
        @param player the player inserting the piece
        @param verbose if True, logs the information
        @return True if insertion was successful
        """
        if verbose:
            print('Inserting into position [{}, {}]'.format(position, self.stacks[position]))

        if self.stacks[position] >= 0:
            self.board[position, self.stacks[position]] = self.players.index(player) + 1
            self.stacks[position] -= 1
            return True
        else:
            return False

    def on_board(self, player_pos, coords):
        """
        Takes in array of coordinates and determines whether player has all pieces.
        @param player_pos position of player to check
        @param coords array of coordinates
        @return True or False
        """
        is_on_board = True

        for coord in coords:
            is_on_board = is_on_board and self.board[coord[0], coord[1]] == player_pos
        
        return is_on_board

    def is_connected(self, player, conns=4):
        """
        Checks diagonals, horizontal and vertical conditions for n number of connected pieces (for player).
        @param player the determinant player
        @param conns number of connections to check for
        @return True if player has conns number of connected pieces
        """
        player_pos = self.players.index(player) + 1
        connections, connected, connection_type = [], None, False

        # Right diag
        for col in range(0, conns):
            for row in reversed(range(conns-1, self.rows)):
                joints = []

                for n in range(conns):
                    joints.append((col+n, row-n))

                if self.on_board(player_pos, joints):
                    connections, connection_type, connected = joints, 'diagonal', True

                if connected:
                    break
    
        # Left diag
        if not connected:
            for col in reversed(range(conns-1, self.columns)):
                for row in reversed(range(conns-1, self.rows)):
                    joints = []

                    for n in range(conns):
                        joints.append((col-n, row-n))

                    if self.on_board(player_pos, joints):
                        connections, connection_type, connected = joints,  'diagonal', True

                    if connected:
                        break

        # Vertical
        if not connected:
            for col in range(self.columns):
                for row in reversed(range(conns-1, self.rows)):
                    joints = []

                    for n in range(conns):
                        joints.append((col, row-n))
                    
                    if self.on_board(player_pos, joints):
                        connections, connection_type, connected = joints, 'vertical', True

                    if connected:
                        break

        # Horizontal
        if not connected:
            for col in range(conns):
                for row in range(self.rows):
                    joints = []

                    for n in range(conns):
                        joints.append((col+n, row))
                    
                    if self.on_board(player_pos, joints):
                        connections, connection_type, connected = joints, 'horizontal', True

                    if connected:
                        break

        return connections, connection_type, connected
    
    def is_tie(self):
        """
        Checks if the board is full (tie game).
        @return True if the game ends in a tie
        """
        is_tie = np.amin(self.board.flatten()) != 0

        if is_tie:
            self.winner = 'None'

        return is_tie

    def reset(self):
        """
        Resets game to initial state.
        """
        self.board = np.zeros((self.columns, self.rows), dtype=int)
        self.stacks = np.ones((self.columns), dtype=int) * self.rows - 1
        self.winner, self.done = None, False

    def step(self, actions, policy=None, verbose=False):
        """
        Gym env-like step method. Takes an in-game step for each player.
        @param actions ordered array of actions for each player to take
        @param policy policy to determine actions if insertion fails
        @return np array of rewards, array of actions, done, info
        """
        if policy is None:
            policy = np.random.randint

        if verbose:
            print('Actions: {}'.format(actions))

        rewards = np.ones((len(self.players))) * -.01

        for index, player in enumerate(self.players):
            inserted = False

            while not inserted:
                if self.insert(actions[index], player, verbose):
                    inserted = True
                else:
                    actions[index] = policy(self.actions)
            
            _conns, _ctype, connected = self.is_connected(player)

            if connected:
                self.winner = player
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
        """
        Provides in-game information including game state and rewards.
        @param rewards rewards from last step
        @return formatted information
        """
        info = 'Info:'
        info += '\nState: {}'.format(self.get_state())
        info += '\nRewards: {}'.format(rewards)

        return info

    def render(self):
        """
        Renders the current game state.
        """
        board, players = '', ['None'] + self.players

        print('~*~ Connect 4 ~*~')
        print('State:', self.get_state())

        for row in range(self.rows):
            for col in range(self.columns):
                board += '| {} |'.format(self.board[col, row])
            board += '\n'

        for i in range(self.columns):
            board += '  {}  '.format(i)
        board += '\n'

        print(board)