from state import State
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

            if action[0] == -1:
                print 'Game is Draw'
                self.finish = True
                break

            next_state = self.state.next_state(action, self.turn)

            opponent_turn = 3 - self.turn
            reward = next_state.rewards[self.turn]
            opponent_reward = next_state.rewards[opponent_turn]
            logging.debug('Reward for ' + str(self.turn) + ': ' + str(reward))
            opponent_reward = opponent_reward - reward
            logging.debug('Reward for ' + str(opponent_turn) + ': ' + str(opponent_reward))

            if self.last_action is not None:
                if self.turn == 1:
                    reflected_previous_state = self.previous_state.reflected_state()
                    reflected_next_state = next_state.reflected_state()

                    self.agents[opponent_turn].remember(reflected_previous_state, self.last_action,
                                                        opponent_reward, reflected_next_state)
                else:
                    self.agents[opponent_turn].remember(self.previous_state, self.last_action,
                                                        opponent_reward, next_state)

            self.last_action = action
            self.previous_state = self.state
            self.state = next_state
            self.turn = opponent_turn
            self.finish = self.state.done()

            if self.test:
                print(str(self.state))
                print ''

        print(str(self.state))