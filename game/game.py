import pygame

from agents import HumanAgent
from agents import ComputerAgent
from gameclient import *
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
        print('Starting game')
        pygame.init()

        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.bind((self.ip_address, 0))
        self.address = self.sock.getsockname()
        print('Running from ' + str(self.sock.getsockname()))
        self.sock.listen(1)


    def _initialize_agent(self):
        pass

    def run(self):
        self._start_game_server()

        human_player = GameClient(HumanAgent(), self.address)
        human_connection, addr = self.sock.accept()

        ai_player = GameClient(ComputerAgent(), self.address)
        ai_connection, addr = self.sock.accept()

        human_player.start()
        ai_player.start()

        human_check_in = human_connection.recv(1024)
        ai_check_in = ai_connection.recv(1024)

        print(human_check_in)
        print(ai_check_in)

        player_connections = [human_connection, ai_connection]

        while not self.state.finish():
            print('Player turn: ' + str(self.turn))
            player_connections[self.turn].send('move your ass')
            action = player_connections[self.turn].recv(1024)
            self.state = self.state.next_state(action)

            reward = self.state.get_reward(self.turn)

            self.current_player.remember(self.state, action, reward)

            self.turn = 3 - self.turn

        print('OUT OF LOOP')
        human_player.keepRunning = False
        ai_player.keepRunning = False

        human_player.join()
        ai_player.join()
        print('ALL PLAYERS FINISH')