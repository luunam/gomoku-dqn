from agents.agent import Agent
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
import numpy as np
import random
from random import randint


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
        self.old_moves = set()

        self.epsilon = 1
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        self.isCrazy = False

    def _build_model(self):
        model = Sequential()
        model.add(Dense(255, input_dim=self.state_size, activation='relu'))
        model.add(Dense(255, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state, last_move):
        if last_move is not None:
            self.old_moves.add(last_move[0] * self.board_size + last_move[1])

        if len(self.old_moves) == self.action_size:
            return -1, -1

        if np.random.rand() <= self.epsilon or self.isCrazy:
            best_action = randint(0, self.action_size-1)
            while best_action in self.old_moves:
                best_action = randint(0, self.action_size-1)

        else:
            self.gamestate = state
            state = state.get_np_value()
            act_value = self.model.predict(state)

            sorted_arg = np.argsort(act_value)[0]
            k = 1
            while k <= self.action_size:
                best_action = sorted_arg[self.action_size - k]
                if best_action not in self.old_moves:
                    self.old_moves.add(best_action)
                    break

                k+=1

        x = best_action / self.board_size
        y = best_action % self.board_size

        # print 'Computer move: ' + str(x) + ' ' + str(y)
        return x, y

    def replay(self, batch_size):
        if batch_size < len(self.memory):
            batch = random.sample(self.memory, batch_size)
        else:
            batch = self.memory

        for state, action, reward, next_state in batch:
            state_np = state.get_np_value()
            next_state_np = next_state.get_np_value()

            target = self.duplicate_model.predict(state_np)

            # print str(target)

            q_value = reward + self.gamma * np.amax(self.duplicate_model.predict(next_state_np))
            action_idx = action[0] * self.board_size + action[1]

            target[0][action_idx] = q_value

            self.model.fit(state_np, target, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        self.duplicate_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def reset(self):
        self.old_moves = set()

    def save(self, name):
        self.model.save_weights(name)

    def brain_wash(self):
        self.isCrazy = True

    def rehab(self):
        self.isCrazy = False
