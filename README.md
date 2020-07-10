# Connect 4

A two-player Connect 4 program that trains by playing matches between two AIs. After training data exists a player can then challenge the AI. Q-Learning is the ML algorithm used to train. Double-Q Learning was tested but appears to have smaller returns and trains a lot slower.

Feel free to make PRs and suggest improvements.

## Instructions:
1. Have Python (and pip) installed (I use Python 3.x)
2. Install the dependencies:
    - `pip install numpy` (for Math stuff)
    - `pip install tqdm` (for progress bar)
    - `pip install random` (for random.choice)
    - `pip install os` (for file system)
    - `pip install re` (for regex)
3. Usage: 
    - `python game.py` (runs program with 10k matches of training data)
    - `python game.py -m 20000 -p False` (trains AI on 20k matches; does not play game after)
    - `python game.py --matches=100000 -play True` (trains AI on 100k matches; plays game after)

## Roadmap
1. ~~Add more documentation to code~~
2. ~~Save AI state to resume training from previous point~~
3. ~~Introduce player vs computer (AI)~~
4. Implement DQN to improve AI performance
5. Create web UI where players can play against Connect4 bot online (and it learns from playing users)