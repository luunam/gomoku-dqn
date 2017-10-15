class Agent:
    def __init__(self, board_size):
        self.gamestate = None
        self.board_size = board_size
        pass

    def act(self, gamestate=None):
        pass

    def replay(self, batch_size):
        pass

    def remember(self):
        pass