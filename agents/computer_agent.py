from agents.agent import Agent
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np


class ComputerAgent(Agent):
    def __init__(self, board_size):
        Agent.__init__(self, board_size)
        self.name = 'computer'
        self.model = self._build_model()
        self.state_size = self.board_size * self.board_size * 3

    def _build_model(self):
        model = Sequential()
        model.add(Dense(255, input_dim=self.state_size))
        model.add(Activation('sigmoid'))
        model.add(Dense(255))
        model.add(Activation('sigmoid'))

        return model

    def _convert_gamestate_to_np(self, state):
        return np.zeros()

    def act(self, gamestate=None):
        self.gamestate = gamestate
        state = self._convert_gamestate_to_np(gamestate)
        act_value = self.model.predict(state)
        return act_value

    def replay(self, batch_size):
        pass

    def remember(self):
        pass