import random


class State:
    def __init__(self, size, board=None):
        self.size = size
        if board is None:
            self.board = [[0 for x in range(self.board_size)] for y in range(self.board_size)]
        else:
            self.board = board

    def next_state(self, action):
        clone_board = self._clone_board()
        clone_board[action[0]][action[1]] = 1
        return State(15, clone_board)

    def get_reward(self, player):
        return 0

    def _clone_board(self):
        to_return = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        for i in range(self.size):
            for j in range(self.size):
                to_return[i][j] = self.board[i][j]

        return to_return

    def finish(self):
        rand = random.randrange(1, 5, 1)
        print('RAND: ' + str(rand))
        if rand == 1:
            return True



