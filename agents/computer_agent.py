from agents.agent import Agent
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np


class ComputerAgent(Agent):
    def __init__(self, board_size, turn):
        Agent.__init__(self, board_size, turn)
        self.name = 'computer'
        self.state_size = self.board_size * self.board_size * 3
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.memory = []


    def _build_model(self):
        model = Sequential()
        model.add(Dense(255, input_dim=self.state_size, activation='sigmoid'))
        model.add(Dense(255, activation='sigmoid'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def _convert_gamestate_to_np(self, state):
        return np.random.random((1, self.state_size))

    def act(self, state=None):
        self.gamestate = state
        state = self._convert_gamestate_to_np(state)

        print(state)
        print(state.shape)
        act_value = self.model.predict(state)
        return act_value

    def replay(self, batch_size):
        pass

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))
        pass