import pygame

from agents import HumanAgent
from agents import ComputerAgent
from gameclient import *
import random


class Game:
    def __init__(self):
        self.players = []
        self.finish = False
        self.ip_address = '127.0.0.1'
        self.turn = 0
        pass

    def run(self):
        print('Starting game')
        pygame.init()

        sock = socket(AF_INET, SOCK_STREAM)
        sock.bind((self.ip_address, 0))
        address = sock.getsockname()
        print('Running from ' + str(sock.getsockname()))
        sock.listen(1)

        human_player = GameClient(HumanAgent(), address)
        human_connection, addr = sock.accept()

        ai_player = GameClient(ComputerAgent(), address)
        ai_connection, addr = sock.accept()

        human_player.start()
        ai_player.start()

        human_check_in = human_connection.recv(1024)
        ai_check_in = ai_connection.recv(1024)

        print(human_check_in)
        print(ai_check_in)
        player_connections = [human_connection, ai_connection]

        while not self.finish:
            print('Player turn: ' + str(self.turn))
            player_connections[self.turn].send('move your ass')
            data = player_connections[self.turn].recv(1024)
            print(data)
            self.turn = (self.turn + 1) % 2
            self.check_finish()

        print('OUT OF LOOP')
        human_player.keepRunning = False
        ai_player.keepRunning = False

        human_player.join()
        ai_player.join()
        print('ALL PLAYERS FINISH')

    def check_finish(self):
        rand = random.randrange(1, 5, 1)
        print('RAND: ' + str(rand))
        if rand == 1:
            self.finish = True