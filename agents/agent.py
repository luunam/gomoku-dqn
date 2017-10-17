class Agent:
    def __init__(self, board_size):
        self.gamestate = None
        self.board_size = board_size
        pass

    def act(self, gamestate, last_move):
        pass

    def replay(self, batch_size):
        pass

    def remember(self, state, action, reward, next_state):
        pass

    def reset(self):
        pass

    def save(self, name):
        pass