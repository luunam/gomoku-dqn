from agents.agent import Agent
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation


class ComputerAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.name = 'computer'
        self.model = self._build_model()
        self.state_size = 765

    def _build_model(self):
        model = Sequential()
        model.add(Dense(255, input_dim=self.state_size))
        model.add(Activation('sigmoid'))
        model.add(Dense(255))
        model.add(Activation('sigmoid'))

        return model

    def move(self, gamestate=None):
        self.gamestate = gamestate
        return {'x': 1, 'y': 1}

    def replay(self, batch_size):
        pass
