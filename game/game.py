from .state import State
from colorama import init
import logging


class Game:
    def __init__(self, agent1, agent2, size=15, test=False):
        self.agents = ['ignore', agent1, agent2]
        self.finish = False
        self.ip_address = '127.0.0.1'
        self.turn = 1
        self.board_size = size
        self.previous_state = State(self.board_size)
        self.state = State(self.board_size)
        self.last_action = None
        self.test = test

        init()

    def _start_game_server(self):
        pass

    def run(self):
        while not self.finish:
            action = self.agents[self.turn].act(self.state)

            if action == -1:
                print('Game is Draw')
                self.finish = True
                break

            if self.state.valid_move(action):
                opponent_turn = 3 - self.turn
                next_state, reward = self.state.next_state(action, self.turn)
                self.agents[self.turn].remember(reward)

            else:
                next_state = self.state
                self.agents[self.turn].remember(-1)
                opponent_turn = self.turn

            self.last_action = action
            self.state = next_state
            self.turn = opponent_turn
            self.finish = self.state.done()

            if self.test:
                print(str(self.state))
                print()

        self.agents[self.turn].remember(-1)
        self.agents[self.turn].observe(self.state)

        print('Winner: ' + str(3 - self.turn))
        print(str(self.state))
