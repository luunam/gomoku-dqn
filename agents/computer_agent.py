from agents.agent import Agent
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np
import random
from random import randint
import math


class ComputerAgent(Agent):
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
        self.gamma = 0.95

        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        self.isCrazy = False

    def _build_model(self):
        model = Sequential()
        model.add(Dense(225, input_dim=self.state_size, activation='sigmoid', kernel_initializer='zeros'))
        model.add(Dense(225, activation='sigmoid', kernel_initializer='zeros'))
        model.add(Dense(225, activation='sigmoid', kernel_initializer='zeros'))
        model.add(Dense(self.action_size, activation='sigmoid', kernel_initializer='zeros'))

        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state, last_move):
        if np.random.rand() <= self.epsilon:
            best_action = randint(0, self.action_size-1)
            while not state.valid_move(best_action):
                best_action = randint(0, self.action_size-1)

        else:
            self.gamestate = state
            best_action, best_action_value = self.get_best_move(self.model, self.gamestate)
            # print "BEST ACTION VALUE: " + str(best_action_value)

        x = best_action / self.board_size
        y = best_action % self.board_size

        return x, y

    def get_best_move(self, model, state):
        state_np = state.get_np_value()
        act_value = model.predict(state_np)

        sorted_arg = np.argsort(act_value)[0]
        k = 1
        while k <= self.action_size:
            best_action = sorted_arg[self.action_size - k]
            if state.valid_move(best_action):
                break

            k += 1

        return best_action, act_value[0][best_action]

    def replay(self, batch_size):
        if batch_size < len(self.memory):
            batch = random.sample(self.memory, batch_size)
        else:
            batch = self.memory

        for state, action, reward, next_state in batch:
            state_np = state.get_np_value()

            target = self.duplicate_model.predict(state_np)

            if next_state.done():
                q_value = reward

            else:
                best_action_idx, best_action_value = self.get_best_move(self.duplicate_model, next_state)
                q_value = reward + self.gamma * best_action_value

            if q_value < 0:
                print 'NEGATIVE Q VALUE: ' + str(q_value)

            action_idx = action[0] * self.board_size + action[1]
            action_value = self.sigmoid(q_value)

            target[0][action_idx] = action_value

            self.model.fit(state_np, target, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        self.duplicate_model.set_weights(self.model.get_weights())

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def save(self, name):
        self.model.save_weights(name)

    def brain_wash(self):
        self.epsilon = 0.4

    def rehab(self):
        self.isCrazy = False

    def load(self, name):
        self.model.load_weights(name)
        self.duplicate_model.load_weights(name)
