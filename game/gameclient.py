from __future__ import print_function
from threading import Thread
from socket import *
import errno


class GameClient(Thread):
    def __init__(self, agent, address):
        Thread.__init__(self)
        self.agent = agent
        self.keep_running = True
        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.connect(address)
        self.sock.setblocking(0)
        self.sock.send(self.agent.name + ' connected')

    def run(self):
        while self.keep_running:
            try:
                data = self.sock.recv(1024)
                move = self.agent.act(data)
                self.sock.send(self.agent.name + ' moved: ' + str(move))
            except error as err:
                if err.errno != errno.EWOULDBLOCK:
                    sys.exit()
