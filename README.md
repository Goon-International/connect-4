# Connect 4

A two-player Connect 4 program that plays matches betweenn two AIs. One imaginary player trains against the other. P1 is the learner. P2 doesn't do much improving for now. Q-Learning is the ML algorithm P1 uses to train. Double-Q Learning was tested but appears to have smaller returns and trains a lot slower. Feel free to make PRs and suggest improvements.

## Instructions:
1. Have Python (and pip) installed (I use Python 3.x)
2. Install the dependencies:
    - `pip install numpy` (for Math stuff)
    - `pip install tqdm` (for progress bar)
3. Usage: `python3 learner.py`

## Roadmap
1. ~~Add more documentation to code~~
2. ~~Save AI state to resume training from previous point~~
3. ~~Introduce player vs computer (AI)~~
4. Speed up mathy operations w/ NumExpr
5. Create web UI where players can play against Connect4 bot online (and it learns from playing users)
