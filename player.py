from __future__ import print_function
from threading import Thread
from socket import *


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