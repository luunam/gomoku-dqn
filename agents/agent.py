class Agent:
    def __init__(self, board_size):
        self.gamestate = None
        self.board_size = board_size
        pass

    def act(self, observation, reward, done):
        pass

    def replay(self, batch_size):
        pass

    def remember(self, reward, next_state, done):
        pass

    def save(self, name):
        pass

    def load(self, name):
        pass

    def observe(self, state):
        pass