from agents.agent import Agent
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np


class ComputerAgent(Agent):
    def __init__(self, board_size, turn):
        Agent.__init__(self, board_size, turn)
        self.name = 'computer'
        self.state_size = self.board_size * self.board_size
        self.action_size = self.board_size * self.board_size
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.memory = []
        self.gamma = 0.95

    def _build_model(self):
        model = Sequential()
        model.add(Dense(25, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state=None):
        self.gamestate = state
        state = state.get_np_value()
        act_value = self.model.predict(state)

        best_action = np.argmax(act_value)
        print 'Best action: ' + str(best_action)
        x = best_action / self.board_size
        y = best_action % self.board_size

        print 'Computer move: ' + str(x) + ' ' + str(y)
        return x, y

    def replay(self, batch_size):
        for state, action, reward, next_state in self.memory:
            state_np = state.get_np_value()
            next_state_np = next_state.get_np_value()

            target = self.model.predict(state_np)
            target[0][action] = reward + self.gamma * np.amax(self.model.predict(next_state_np))

            self.model.fit(state, target, epoch=1, verbose=0)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))