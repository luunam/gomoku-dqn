from agents.agent import Agent
import torch

import copy
import random
import logging
from .dqn_net import DQNNet

from game.state import State


class DQNAgent(Agent):
    def __init__(self, board_size):
        Agent.__init__(self, board_size)

        self.learning_rate = 0.05

        self.model = DQNNet()
        self.duplicate_model = copy.deepcopy(self.model)

        self.memory = []
        self.gamma = 0.5

        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        self.last_action = None
        self.last_reward = None
        self.last_state = None

    def act(self, observation: State, reward: int, done: bool) -> int:
        if done:
            # If done is true that means we lose the game
            self.remember(-1, observation, done)

            # Whatever we return here is not important anymore
            return -1
        else:
            self.remember(reward, observation, done)

            _, best_move = self.find_best_move(observation)

            self.last_state = observation
            self.last_action = best_move

            return best_move

    def remember(self, reward: int, next_state: State, done: bool):
        if self.last_state is not None and self.last_action is not None:
            if len(self.memory) > 1000:
                self.memory = random.sample(self.memory, 500)

            self.memory.append((self.last_state, self.last_action, reward, next_state, done))

    def find_best_move(self, state: State, model: DQNNet=None) -> (int, int):
        """
        Find the best legal move given a state of the game

        :param state:
        :param model: The model that we want to use to find the best move, default to be self.model

        :return: The best move that is legal
                 If the best move the model can find is an illegal move (i.e. moving to a square that
                 is already occupied), then that move is ignored and we will find the first best move that is legal
        """
        if model is None:
            model = self.model

        act_value = model.predict(state)
        max_value, max_value_idx = torch.max(act_value, 1)
        max_value = max_value.data[0]
        max_value_idx = max_value_idx.data[0]
        # print('Best move: {}, ({}, {})'.format(max_value, max_value_idx // 15, max_value_idx % 15))
        return max_value, max_value_idx

    def replay(self, batch_size: int):
        if batch_size < len(self.memory):
            batch = random.sample(self.memory, batch_size)
        else:
            batch = self.memory

        logging.debug('Replaying: ')

        for state, action, reward, next_state, done in batch:
            self.log_batch(state, action, reward, next_state)

            target = self.duplicate_model.predict(state)

            if done:
                q_value = reward
            else:
                _, best_action_value = self.find_best_move(next_state, self.duplicate_model)
                q_value = reward + self.gamma * best_action_value

            target[0, action] = q_value

            self.model.fit_single_data(DQNNet.state_to_tensor(state), target.data)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        self.duplicate_model = copy.deepcopy(self.model)

    @staticmethod
    def log_batch(state, action, reward, next_state):
        logging.debug('State: ' + '\n' + str(state))
        logging.debug('Action: ' + str(action))
        logging.debug('Reward: ' + str(reward))
        logging.debug('Next state: ' + '\n' + str(next_state) + '\n')

    def print_memory(self):
        """
        Print memory of this agent, for debugging purpose only
        """
        print('Memory size {}'.format(len(self.memory)))
        for state, action, reward, next_state, done in self.memory:
            print('State: ' + '\n' + str(state))
            x = action // 15
            y = action % 15
            print('Action: ' + '\n' + str(x) + ' ' + str(y))
            print('Reward: ' + '\n' + str(reward))
            print('Next state: ' + '\n' + str(next_state))

    def save(self, name: str):
        torch.save(self.model.state_dict(), name)

    def load(self, name: str):
        self.model.load_state_dict(torch.load(name))
        self.duplicate_model = copy.deepcopy(self.model)


class Memory:
    def __init__(self):
        pass



