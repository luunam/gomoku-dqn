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

        self.finish = False
        self.winner = 0

    def next_state(self, action, turn):
        clone_board = self._clone_board()
        clone_board[action[0]][action[1]] = turn
        return State(15, clone_board)

    def get_reward(self, turn):
        accumulate = ''
        result = {
            'open_three': 0,
            'four': 0,
            'open_four': 0,
            'five': 0
        }

        # Traverse row
        for i in range(self.size):
            for j in range(self.size):
                accumulate = self.evaluate(i, j, accumulate, turn, result)

            accumulate = ''

        if self.finish:
            return 1000

        # Traverse column
        accumulate = ' '
        for j in range(self.size):
            for i in range(self.size):
                accumulate = self.evaluate(i, j, accumulate, turn, result)

            accumulate = ''

        if self.finish:
            return 1000

        # Traverse sum diagonal (diagonal that i+j are equal)
        for sum in range(0, 2*(self.size-1)):
            x_max = min(sum, self.size - 1)
            x_min = max(0, sum - self.size + 1)
            for x in range(x_min, x_max+1):
                y = sum - x
                accumulate = self.evaluate(x, y, accumulate, turn, result)

            accumulate = ''

        if self.finish:
            return 1000

        # Traverse diff diagonal:
        for diff in range(-(self.size - 1), self.size):
            x_max = min(self.size-1, self.size + diff - 1)
            x_min = max(0, diff)
            for x in range(x_min, x_max+1):
                y = x - diff
                accumulate = self.evaluate(x, y, accumulate, turn, result)

            accumulate = ''

        if self.finish:
            return 1000

        return result['open_three'] + result['four'] + 5 * result['open_four']

    def evaluate(self, i, j, accumulate, turn, result):
        if len(accumulate) < 6:
            if self.board[i][j] == turn:
                accumulate += 'x'
            elif self.board[i][j] == 0:
                accumulate += ' '
            else:
                accumulate += 'y'

        if len(accumulate) == 6:
            if accumulate == ' xx x ' or accumulate == ' x xx ' or accumulate[0:5] == ' xxx ' \
                    or accumulate[1:6] == ' xxx ':
                result['open_three'] += 1

            if accumulate == ' xxxx ':
                result['open_four'] += 1
                result['four'] -= 2

            five = [accumulate[0:5], accumulate[1:6]]
            for tmp in five:
                if tmp == 'xxx x' or tmp == 'xx xx' or tmp == 'x xxx' or tmp == ' xxxx' or tmp == 'xxxx ':
                    result['four'] += 1

                if tmp == 'xxxxx':
                    result['five'] += 1
                    self.finish = True
                    self.winner = turn
                    return accumulate

            accumulate = accumulate[1:6]

        return accumulate

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

    def done(self):
        return self.finish



