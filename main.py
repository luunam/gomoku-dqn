from __future__ import print_function
from game.game import Game

episodes = 2000

def train(self):
    for _ in range(episodes):
        game = Game()
        game.run()


if __name__ == "__main__":
    game = Game()
    game.run()
