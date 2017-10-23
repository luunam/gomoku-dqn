from __future__ import print_function
from game.game import Game
from agents import HumanAgent, ComputerAgent
import argparse

EPISODES = 10000
SIZE = 15


def train():
    agent1 = ComputerAgent(SIZE)
    agent2 = ComputerAgent(SIZE)

    for i in range(EPISODES):
        print("EPISODE: " + str(i))
        new_game = Game(agent1, agent2)
        new_game.run()

        agent1.replay(100)
        agent2.replay(100)

    agent1.save('./trained/agent1.h5')
    agent2.save('./trained/agent2.h5')


def test():
    agent1 = HumanAgent(SIZE)
    agent2 = ComputerAgent(SIZE)

    agent1.load('./trained/agent1.h5')
    agent2.load('./trained/agent2.h5')

    agent1.epsilon = 0
    agent2.epsilon = 0

    new_game = Game(agent1, agent2)
    new_game.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command, either train or test')
    parser.add_argument('command', type=str)

    args = parser.parse_args()

    command = args.command

    if command == 'train':
        train()

    if command == 'test':
        test()