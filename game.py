from __future__ import print_function
import pygame
from socket import *
import time

from threading import Thread

# Creates a list containing 5 lists, each of 8 items, all set to 0
w, h = 4, 4
board = [[' ' for x in range(w)] for y in range(h)]

TCP_IP = '127.0.0.1'
TCP_PORT = 1604


class Player(Thread):
    def __init__(self, agent, address):
        Thread.__init__(self)
        self.agent = agent
        self.keepRunning = True
        self.sock = socket(AF_INET, SOCK_STREAM)

        self.sock.connect(address)
        self.sock.send(self.agent.name + ' connected')

    def run(self):
        while self.keepRunning:
            print('running in thread ' + self.agent.name)


class Agent:
    def __init__(self):
        self.gamestate = None
        pass

    def move(self, gamestate=None):
        pass


class ComputerAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.name = 'computer'

    def move(self, gamestate=None):
        self.gamestate = gamestate
        move = {x: 1, y: 1}
        print(move)
        event = pygame.event.Event(pygame.USEREVENT, move)
        pygame.event.post(event)


class HumanAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.name = 'human'

    def move(self, gamestate=None):
        print('Human move')
        while True:
            for event in pygame.event.get():
                print(event)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    print('Done')
                    event = pygame.event.Event(pygame.USEREVENT, {})
                    pygame.event.post(event)


def print_board(board):
    """
    @:param board:
    :return:
    """
    for i in range(h):
        for j in range(w):
            print('(' + board[i][j] + ')', end='')
        print()


def main():
    print('Starting game')
    pygame.init()

    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind((TCP_IP, 0))
    address = sock.getsockname()
    print('Running from ' + str(sock.getsockname()))
    sock.listen(1)

    human_player = Player(HumanAgent(), address)
    human_connection, addr = sock.accept()

    ai_player = Player(ComputerAgent(), address)
    ai_connection, addr = sock.accept()

    human_player.start()
    ai_player.start()

    while 1:
        data = human_connection.recv(1024)
        print(data)
        data = ai_connection.recv(1024)
        print(data)

        human_player.keepRunning = False
        ai_player.keepRunning = False

        human_player.join()
        ai_player.join()
        print ('MAIN THREAD IS DONE')
        break

if __name__ == "__main__":
    main()