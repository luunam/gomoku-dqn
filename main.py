from __future__ import print_function
from game.game import Game
from agents import HumanAgent, ComputerAgent

EPISODES = 1000
SIZE = 15


def train():
    agent1 = ComputerAgent(SIZE)
    agent2 = ComputerAgent(SIZE)

    for i in range(EPISODES):
        print("EPISODE: " + str(i))
        new_game = Game(agent1, agent2)
        new_game.run()

        agent1.replay(10)
        agent2.replay(10)

        agent1.reset()
        agent2.reset()

    agent1.save('./trained/agent1.h5')
    agent2.save('./trained/agent2.h5')


if __name__ == "__main__":
    # human_agent = HumanAgent(SIZE)
    # computer_agent = ComputerAgent(SIZE)
    # game = Game(human_agent, computer_agent)
    # game.run()

    train()
