from __future__ import print_function
from game.game import Game
from agents import HumanAgent, DQNAgent, MinimaxAgent
from benchmark import Benchmarker
import argparse
import logging
import time
import os


EPISODES = 25000
SIZE = 15
BATCH_SIZE = 100


def train():
    agent1 = DQNAgent(SIZE)
    agent2 = DQNAgent(SIZE)
    print('Train with: ' + str(EPISODES) + ' episodes and batch size: ' + str(BATCH_SIZE))

    player_train = 0

    if not os.path.exists('./trained'):
        os.makedirs('./trained')

    try:
        start = time.time()
        for episode in range(EPISODES):
            print("Episode: " + str(episode))
            new_game = Game(agent1, agent2, test=False)
            new_game.run()

            if player_train == 0:
                agent1.replay(BATCH_SIZE)
            else:
                agent2.replay(BATCH_SIZE)

            file_name1 = 'agent1.h5'
            file_name2 = 'agent2.h5'

            if episode % 50 == 0:
                print('Time elapsed: ' + str(time.time() - start))
                start = time.time()
                player_train = 1 - player_train
                if os.path.isfile('./trained/agent1.h5'):
                    os.remove('./trained/agent1.h5')
                if os.path.isfile('./trained/agent2.h5'):
                    os.remove('./trained/agent2.h5')

                agent1.save('./trained/' + file_name1)
                agent2.save('./trained/' + file_name2)

        agent1.save('./trained/agent1.h5')
        agent2.save('./trained/agent2.h5')

    except KeyboardInterrupt:
        print('Key board interrupt, emergency save')
        agent1.save('./trained/agent1_interrupt.h5')
        agent2.save('./trained/agent2_interrupt.h5')


def test():
    human_agent = HumanAgent(SIZE)
    dqn_agent = DQNAgent(SIZE)
    dqn_agent_2 = DQNAgent(SIZE)

    human_agent.epsilon = 0
    dqn_agent.epsilon = 0
    dqn_agent_2.epsilon = 0

    new_game = Game(dqn_agent, human_agent, SIZE, True)
    new_game.run()

    print(len(dqn_agent.memory.ephemeral_storage))

    new_game = Game(dqn_agent, human_agent, SIZE, True)
    new_game.run()

    print(len(dqn_agent.memory.ephemeral_storage))

    new_game = Game(dqn_agent, human_agent, SIZE, True)
    new_game.run()

    print(len(dqn_agent.memory.ephemeral_storage))

    dqn_agent.print_memory()


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
