# Connect 4

A two-player Connect 4 program that trains by playing matches between two AIs. After training data exists a player can then challenge the AI. Q-Learning is the ML algorithm used to train. Double-Q Learning was tested but appears to have smaller returns and trains a lot slower.

Feel free to make PRs and suggest improvements.

## Instructions:
1. Have Python (and pip) installed (I use Python 3.x)
2. Install the dependencies:
    - `pip install numpy` (for Math stuff)
    - `pip install numexpr` (speed up mathy operations)
    - `pip install tqdm` (for progress bar)
    - `pip install random` (for random.choice)
    - `pip install os` (for file system)
    - `pip install re` (for regex)
3. Usage: `python3 game.py`

## Roadmap
1. ~~Add more documentation to code~~
2. ~~Save AI state to resume training from previous point~~
3. ~~Introduce player vs computer (AI)~~
4. Create web UI where players can play against Connect4 bot online (and it learns from playing users)

## Benchmarks
    - Playing 10,000 matches takes 2m5s
    - Playing 15,000 matches takes 2m39s
    - Playing 25,000 matches takes 4m30s
    - Playing 50,000 matches takes 9m16s