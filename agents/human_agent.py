from agents.agent import Agent


class HumanAgent(Agent):
    def __init__(self, board_size):
        Agent.__init__(self, board_size)
        self.name = 'human'

    def act(self, gamestate=None):
        print('Human move')
        x = raw_input('x: ')
        y = raw_input('y: ')

        return {'x': x, 'y': y}

    def replay(self, batch_size):
        pass
