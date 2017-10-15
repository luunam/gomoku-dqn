import pygame

from agents import HumanAgent
from agents import ComputerAgent
from state import State


class Game:
    def __init__(self):
        self.agents = []
        self.finish = False
        self.ip_address = '127.0.0.1'
        self.turn = 1
        self.board_size = 15
        self.state = State(self.board_size)

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

        while not self.state.finish():
            print('Player turn: ' + str(self.turn))

            action = self.agents[self.turn].act(self.state)

            self.state = self.state.next_state(action)
            reward = self.state.get_reward(self.turn)

            self.agents[self.turn].remember(self.state, action, reward)

            self.turn = 3 - self.turn
            self.finish = self.state.finish()