from __future__ import print_function
from threading import Thread
from socket import *
import errno


class GameClient(Thread):
    def __init__(self, agent, address):
        Thread.__init__(self)
        self.agent = agent
        self.keepRunning = True
        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.connect(address)
        self.sock.setblocking(0)
        self.sock.send(self.agent.name + ' connected')

    def run(self):
        while self.keepRunning:
            try:
                data = self.sock.recv(1024)
                print(data)
                self.sock.send(self.agent.name + ' moved')
            except error as err:
                if err.errno != errno.EWOULDBLOCK:
                    sys.exit()
