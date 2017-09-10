import pygame

from agents import HumanAgent
from agents import ComputerAgent
from player import *
import time

class Game:
    def __init__(self):
        self.players = []
        self.finish = False
        self.ip_address = '127.0.0.1'
        pass

    def run(self):
        print('Starting game')
        pygame.init()

        sock = socket(AF_INET, SOCK_STREAM)
        sock.bind((self.ip_address, 0))
        address = sock.getsockname()
        print('Running from ' + str(sock.getsockname()))
        sock.listen(1)

        human_player = Player(HumanAgent(), address)
        human_connection, addr = sock.accept()

        ai_player = Player(ComputerAgent(), address)
        ai_connection, addr = sock.accept()

        human_player.start()
        ai_player.start()

        while not self.finish:
            data = human_connection.recv(1024)
            print(data)
            data = ai_connection.recv(1024)
            print(data)

            time.sleep(3)
            human_player.keepRunning = False
            ai_player.keepRunning = False

            human_player.join()
            ai_player.join()
            print('MAIN THREAD IS DONE')
            break