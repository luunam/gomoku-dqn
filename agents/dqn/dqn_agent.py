from agents.agent import Agent
import torch
from torch.autograd import Variable

import copy
import random
import logging
from .dqn_net import DQNNet

from game.state import State
from .memory import Memory


class DQNAgent(Agent):
    def __init__(self, board_size):
        Agent.__init__(self, board_size)

        self.model = DQNNet()
        self.duplicate_model = copy.deepcopy(self.model)

        self.memory = Memory()
        self.gamma = 0.5

        self.epsilon = 1
        self.epsilon_decay = 0.99
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
            self.memory.queue((self.last_state, self.last_action, reward, next_state, done))

    def find_best_move(self, state: State, model: DQNNet=None) -> (float, int):
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

        if random.uniform(0, 1) < self.epsilon:
            return self.find_random_move(act_value)

        max_value = max_value.data[0]
        max_value_idx = max_value_idx.data[0]
        # print('Best move: {}, ({}, {})'.format(max_value, max_value_idx // 15, max_value_idx % 15))
        return max_value, max_value_idx

    def find_random_move(self, act_value: Variable) -> (float, int):
        min_value, _ = torch.min(act_value, 1)
        min_value = min_value.data[0]
        while True:
            random_move = random.randint(0, 224)
            if act_value[0, random_move].data[0] != min_value:
                return act_value[0, random_move].data[0], random_move

    def replay(self, batch_size: int):
        batch = self.memory.sample(batch_size)

        logging.debug('Replaying: ')
        batch_error = 0
        for state, action, reward, next_state, done in batch:
            self.log_batch(state, action, reward, next_state)

            target = self.model.predict(state)

            if done:
                q_value = reward
            else:
                _, best_action_value = self.find_best_move(next_state)
                q_value = reward + self.gamma * best_action_value

            target[0, action] = q_value

            batch_error += self.duplicate_model.fit_single_data(DQNNet.state_to_tensor(state), target.data)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        print('Batch average error: {}'.format(batch_error // batch_size))

        self.model = copy.deepcopy(self.duplicate_model)

    @staticmethod
    def log_batch(state, action, reward, next_state):
        logging.debug('State: ' + '\n' + str(state))
        logging.debug('Action: ' + str(action))
        logging.debug('Reward: ' + str(reward))
        logging.debug('Next state: ' + '\n' + str(next_state) + '\n')

    def print_memory(self):
        """
        Print memory.py of this agent, for debugging purpose only
        """
        print('Memory size {}'.format(len(self.memory)))
        print(self.memory)

    def save(self, name: str):
        torch.save(self.model.state_dict(), name)

    def load(self, name: str):
        self.model.load_state_dict(torch.load(name))
        self.duplicate_model = copy.deepcopy(self.model)
