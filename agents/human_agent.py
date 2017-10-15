from agents.agent import Agent


class HumanAgent(Agent):
    def __init__(self, board_size, turn):
        Agent.__init__(self, board_size, turn)
        self.name = 'human'

    def act(self, state=None):
        x = raw_input('x: ')
        y = raw_input('y: ')

        print 'Human move: ' + str(x) + ' ' + str(y)
        return int(x), int(y)

    def replay(self, batch_size):
        pass
