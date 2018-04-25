from agents.agent import Agent


class HumanAgent(Agent):
    def __init__(self, board_size):
        Agent.__init__(self, board_size)
        self.name = 'human'

    def act(self, state, reward, done):
        x = input('x: ')
        y = input('y: ')

        print('Human move: ' + str(x) + ' ' + str(y))
        return int(x) * self.board_size + int(y)
