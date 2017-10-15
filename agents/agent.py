class Agent:
    def __init__(self, board_size, turn):
        self.gamestate = None
        self.board_size = board_size
        self.turn = turn
        pass

    def act(self, gamestate=None):
        pass

    def replay(self, batch_size):
        pass

    def remember(self, state, action, reward, next_state):
        pass