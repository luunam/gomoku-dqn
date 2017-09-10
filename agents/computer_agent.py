from agents.agent import Agent


class ComputerAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.name = 'computer'

    def move(self, gamestate=None):
        self.gamestate = gamestate
        return {'x': 1, 'y': 1}
