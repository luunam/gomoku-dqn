import pygame

from agents import HumanAgent
from agents import ComputerAgent
from state import State
from colorama import init

class Game:
    def __init__(self):
        self.agents = []
        self.finish = False
        self.ip_address = '127.0.0.1'
        self.turn = 1
        self.board_size = 15
        self.state = State(self.board_size)
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

    def _initialize_agent(self):
        pass

    def run(self):
        human_agent = HumanAgent(self.board_size, 1)
        computer_agent = ComputerAgent(self.board_size, 2)

        # add 'ignore' to the list because self.turn starts with 1
        self.agents = ['ignore', computer_agent, human_agent]

        while not self.state.done():
            self.state.print_state()
            print('Player turn: ' + str(self.turn))

            action = self.agents[self.turn].act(self.state)
            next_state = self.state.next_state(action, self.turn)

            self.state = next_state

            opponent_turn = 3 - self.turn
            reward = self.state.get_reward(self.turn)
            opponent_reward = self.state.get_reward(opponent_turn)

            reward = reward - opponent_reward
            print 'reward: ' + str(reward)
            self.agents[self.turn].remember(self.state, action, reward, next_state)

            self.turn = opponent_turn
            self.finish = self.state.done()