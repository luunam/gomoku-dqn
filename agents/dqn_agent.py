from agents.agent import Agent
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np
import random
from random import randint
import math
import logging
from game import State
import time


class DQNAgent(Agent):
    def __init__(self, board_size):
        Agent.__init__(self, board_size)
        self.name = 'computer'
        self.state_size = self.board_size * self.board_size
        self.action_size = self.board_size * self.board_size

        self.learning_rate = 0.05
        self.model = self._build_model()
        self.duplicate_model = self._build_model()

        self.duplicate_model.set_weights(self.model.get_weights())

        self.memory = []
        self.gamma = 0.5

        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.previous_state = None
        self.last_action = None
        self.last_reward = None

    def _build_model(self):
        model = Sequential()
        model.add(Dense(225, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(225, activation='tanh'))
        model.add(Dense(225, activation='tanh'))
        model.add(Dense(self.action_size, activation='sigmoid'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state: 'State') -> int:
        # We will move randomly sometime to break out of local minimum
        if np.random.rand() <= self.epsilon:
            best_action = randint(0, self.action_size-1)
            while not state.valid_move(best_action):
                best_action = randint(0, self.action_size-1)

        else:
            self.gamestate = state
            best_action, best_action_value = self.get_best_move(self.model, self.gamestate)

        if self.previous_state is not None:
            self.memory.append((self.previous_state, self.last_action, self.last_reward, state))
        self.previous_state = state
        self.last_action = best_action

        return best_action

    def get_best_move(self, model, state):
        state_np = state.get_np_value()
        act_value = model.predict(state_np)
        logging.debug("Act value: " + '\n' + str(act_value))

        sorted_arg = np.argsort(act_value)[0]
        k = 1
        # For early stage of the program, the best move may be the move that is already taken
        while k <= self.action_size:
            best_action = sorted_arg[self.action_size - k]
            if state.valid_move(best_action):
                break

            k += 1

        logging.debug('k: ' + str(k))
        logging.debug('Computer move: ' + str(best_action))
        return best_action, act_value[0][best_action]

    def replay(self, batch_size):
        if batch_size < len(self.memory):
            batch = random.sample(self.memory, batch_size)
        else:
            batch = self.memory

        logging.debug('Replaying: ')

        for state, action, reward, next_state in batch:
            logging.debug('State: ' + '\n' + str(state))
            logging.debug('Action: ' + str(action))
            logging.debug('Reward: ' + str(reward))
            logging.debug('Next state: ' + '\n' + str(next_state) + '\n')

            state_np = state.get_np_value()

            target = self.duplicate_model.predict(state_np)

            if next_state.done():
                q_value = reward

            else:
                best_action_idx, best_action_value = self.get_best_move(self.duplicate_model, next_state)
                q_value = reward + self.gamma * best_action_value

            action_idx = action

            logging.debug('Action value: ' + str(q_value))
            target[0][action_idx] = q_value

            for already_move in state.moves:
                target[0][already_move] = 0.5

            logging.debug('Target: ' + '\n' + str(target))

            self.model.fit(state_np, target, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        if len(self.memory) > 750:
            print('Memory is too large, cleaning')
            self.memory = batch

        self.duplicate_model.set_weights(self.model.get_weights())

    def print_memory(self):
        # Print out memory, for debugging purpose
        for state, action, reward, next_state in self.memory:
            print('State: ' + '\n' + str(state))
            x = action // state.size
            y = action % state.size
            print('Action: ' + '\n' + str(x) + ' ' + str(y))
            print('Reward: ' + '\n' + str(reward))
            print('Next state: ' + '\n' + str(next_state))

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x/5.0))

    def remember(self, reward: int):
        self.last_reward = reward

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)
        self.duplicate_model.load_weights(name)
        logging.debug('Model weights: ')
        logging.debug('\n' + str(self.model.get_weights()))
