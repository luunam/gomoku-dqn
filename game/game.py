import pygame

from agents import HumanAgent
from agents import ComputerAgent
from state import State
from colorama import init

class Game:
    def __init__(self, agent1, agent2, size=15):
        self.agents = ['ignore', agent1, agent2]
        self.finish = False
        self.ip_address = '127.0.0.1'
        self.turn = 1
        self.board_size = size
        self.state = State(self.board_size)
        self.last_move = (-1,-1)
        init()

    def _start_game_server(self):
        pass
        # pygame.init()
        #
        # self.sock = socket(AF_INET, SOCK_STREAM)
        # self.sock.bind((self.ip_address, 0))
        # self.address = self.sock.getsockname()
        # print('Running from ' + str(self.sock.getsockname()))
        # self.sock.listen(1)

    def run(self):
        while not self.finish:
            action = self.agents[self.turn].act(self.state, self.last_move)

            if action[0] == -1:
                print 'Game is Draw'
                self.finish = True
                break

            self.last_move = action
            next_state = self.state.next_state(action, self.turn)

            self.state = next_state

            opponent_turn = 3 - self.turn
            reward = self.state.get_reward(self.turn)
            opponent_reward = self.state.get_reward(opponent_turn)

            reward = reward - opponent_reward
            # print 'reward: ' + str(reward)
            self.agents[self.turn].remember(self.state, action, reward, next_state)

            self.turn = opponent_turn
            self.finish = self.state.done()

        self.state.print_state()