from __future__ import print_function
from game.game import Game
from agents import HumanAgent, ComputerAgent
import argparse
import logging

EPISODES = 2000000
SIZE = 15
BATCH_SIZE = 225

conf = {
    'debug': False
}


def train():
    agent = ComputerAgent(SIZE)
    try:
        for i in range(EPISODES):
            print("Episode: " + str(i))
            new_game = Game(agent, agent)
            new_game.run()

            agent.replay(BATCH_SIZE)

        agent.save('./trained/agent.h5')

    except KeyboardInterrupt:
        print('Key board interrupt, emergency save')
        agent.save('./trained/agent_interrupt.h5')


def test():
    human_agent = HumanAgent(SIZE)
    agent2 = ComputerAgent(SIZE)

    human_agent.load('./trained/agent1.h5')
    # agent2.load('./trained/agent.h5')

    human_agent.epsilon = 0
    agent2.epsilon = 0

    new_game = Game(human_agent, agent2, SIZE, True)
    new_game.run()

    agent2.replay(BATCH_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command, either train or test')
    parser.add_argument('command', type=str)
    parser.add_argument('--log', type=str)
    args = parser.parse_args()

    command = args.command

    if args.log:
        print('Saving log to: ' + args.log)
        logging.basicConfig(filename=args.log,
                            level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    if command == 'train':
        train()

    if command == 'test':
        test()