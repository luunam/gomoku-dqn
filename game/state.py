import random
import sys
import numpy as np
from colorama import Fore, Style

class State:
    def __init__(self, size, board=None):
        self.size = size

        if board is None:
            self.board = [[0 for x in range(self.size)] for y in range(self.size)]
        else:
            self.board = board

    def next_state(self, action, turn):
        clone_board = self._clone_board()
        clone_board[action[0]][action[1]] = turn
        return State(15, clone_board)

    def get_reward(self, player):
        return 0

    def _clone_board(self):
        to_return = [[0 for _ in range(self.size)] for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                to_return[i][j] = self.board[i][j]

        return to_return

    def get_np_value(self):
        return np.asarray(self.board).reshape(1, self.size*self.size)

    def print_state(self):
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i][j]
                to_print = str(value)
                if value == 1:
                    to_print = Fore.GREEN + str(value) + Style.RESET_ALL
                if value == 2:
                    to_print = Fore.RED + str(value) + Style.RESET_ALL

                sys.stdout.write(to_print + ' ')

            sys.stdout.write('\n')

    def finish(self):
        return False



