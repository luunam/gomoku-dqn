from __future__ import print_function
from game.game import Game
from agents import HumanAgent, ComputerAgent
import argparse

EPISODES = 1000
SIZE = 15


def train():
    agent1 = ComputerAgent(SIZE)
    agent2 = ComputerAgent(SIZE)

    for i in range(EPISODES):
        print("EPISODE: " + str(i))
        new_game = Game(agent1, agent2)
        new_game.run()

        agent1.replay(500)
        agent2.replay(500)

        if i % 200 == 0:
            agent2.brain_wash()
        if i % 400 == 0:
            agent1.brain_wash()

    agent1.save('./trained/agent1.h5')
    agent2.save('./trained/agent2.h5')


if __name__ == "__main__":
    parser = argparse.ArgumentParser

    parser.add_argument('option', type=str)
    args = parser.parse_args()


    # human_agent = HumanAgent(SIZE)
    # computer_agent = ComputerAgent(SIZE)
    # game = Game(human_agent, computer_agent)
    # game.run()

    train()
