from agents.agent import Agent


class HumanAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.name = 'human'

    def move(self, gamestate=None):
        print('Human move')
        x = raw_input('x: ')
        y = raw_input('y: ')

        return {'x': x, 'y': y}

    def replay(self, batch_size):
        pass
