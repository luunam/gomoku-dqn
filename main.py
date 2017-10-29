from __future__ import print_function
from game.game import Game
from agents import HumanAgent, DQNAgent, MinimaxAgent
from benchmark import Benchmarker
import argparse
import logging
import time


EPISODES = 25000
SIZE = 15
BATCH_SIZE = 100


def train():
    agent = DQNAgent(SIZE)
    benchmarker = Benchmarker()
    print('Train with: ' + str(EPISODES) + ' episodes and batch size: ' + str(BATCH_SIZE))
    try:
        start = time.time()
        for episode in range(EPISODES):
            print("Episode: " + str(episode))
            new_game = Game(agent, agent)
            new_game.run()

            agent.replay(BATCH_SIZE)

            file_name = 'agent_' + str(episode) + '.h5'

            if episode % 500 == 0:
                print('Time elapsed: ' + str(time.time() - start))
                start = time.time()
                agent.save('./trained/' + file_name)

            if episode % 100 == 0:
                score = benchmarker.rate(agent)
                print('Rate: ' + str(score))

        agent.save('./trained/agent.h5')

    except KeyboardInterrupt:
        print('Key board interrupt, emergency save')
        agent.save('./trained/agent_interrupt.h5')


def test():
    human_agent = HumanAgent(SIZE)
    dqn_agent = DQNAgent(SIZE)
    minimax_agent = MinimaxAgent(SIZE)

    benchmarker = Benchmarker()
    rate = benchmarker.rate(minimax_agent)

    print('Rate: ' + str(rate))
    # dqn_agent.load('./trained/agent.h5')
    #
    # human_agent.epsilon = 0
    # dqn_agent.epsilon = 0
    #
    # new_game = Game(human_agent, dqn_agent, SIZE, True)
    # new_game.run()

    # agent2.replay(BATCH_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Command, either train or test')
    parser.add_argument('command', type=str)
    parser.add_argument('-l', '--log', type=str)
    parser.add_argument('-e', '--episodes', type=int)
    parser.add_argument('-b', '--batch_size', type=int)

    args = parser.parse_args()

    command = args.command

    if args.log:
        print('Saving log to: ' + args.log)
        logging.basicConfig(filename=args.log,
                            level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    if args.episodes:
        EPISODES = args.episodes

    if args.batch_size:
        BATCH_SIZE = args.batch_size

    if command == 'train':
        train()

    if command == 'test':
        test()
