import random
import sys
import numpy as np
from colorama import Fore, Style
import logging


class State:
    def __init__(self, size, board=None):
        self.size = size
        self.action_size = self.size * self.size
        if board is None:
            self.board = [[0 for x in range(self.size)] for y in range(self.size)]
        else:
            self.board = board

        self.finish = False
        self.winner = 0
        self.rewards = ['ignore', 0, 0]
        self.occupied = 0

        # Player's turn that does the last move
        self.turn = None
        self.win = False

    def next_state(self, action, turn):
        clone_board = self._clone_board()
        clone_board[action[0]][action[1]] = turn
        self.occupied += 1

        next_state = State(15, clone_board)
        next_state.turn = turn

        next_state.rewards[turn] = next_state.get_reward(turn)
        next_state.rewards[3-turn] = next_state.get_reward(3-turn)

        if self.occupied == self.action_size:
            next_state.finish = True

        next_state.occupied = self.occupied

        return next_state

    def get_reward(self, turn):
        result = self.inspect(turn)
        logging.debug('Result for ' + str(turn) + ': ' + str(result))

        if self.finish:
            if self.winner == turn:
                return 20
            else:
                return 0

        result['two'] = max(0, result['two'])
        result['three'] = max(0, result['three'])
        reward = 0.5 * result['three'] + 2 * result['open_three'] + 3 * result['four'] + 10 * result['open_four']

        if reward == 0:
            return 0.1

        return reward

    def inspect(self, turn):
        accumulate = ''
        result = {
            'three': 0,
            'open_three': 0,
            'four': 0,
            'open_four': 0,
            'five': 0,
            'two': 0
        }

        # Traverse row
        for i in range(self.size):
            for j in range(self.size):
                accumulate = self.evaluate(i, j, accumulate, turn, result)

            accumulate = ''

        if self.finish:
            result['five'] = 1
            return result

        # Traverse column
        accumulate = ' '
        for j in range(self.size):
            for i in range(self.size):
                accumulate = self.evaluate(i, j, accumulate, turn, result)

            accumulate = ''

        if self.finish:
            result['five'] = 1
            return result

        # Traverse sum diagonal (diagonal that i+j are equal)
        for sum_indices in range(0, 2*self.size - 1):
            x_max = min(sum_indices, self.size - 1)
            x_min = max(0, sum_indices - self.size + 1)
            for x in range(x_min, x_max+1):
                y = sum_indices - x
                accumulate = self.evaluate(x, y, accumulate, turn, result)

            accumulate = ''

        if self.finish:
            result['five'] = 1
            return result

        # Traverse diff diagonal:
        for diff in range(-(self.size - 1), self.size):
            x_max = min(self.size-1, self.size + diff - 1)
            x_min = max(0, diff)
            for x in range(x_min, x_max+1):
                y = x - diff
                accumulate = self.evaluate(x, y, accumulate, turn, result)

            accumulate = ''

        if self.finish:
            result['five'] = 1
            return result

        return result

    def evaluate(self, i, j, accumulate, turn, result):
        if self.board[i][j] == turn:
            accumulate += 'x'
        elif self.board[i][j] == 0:
            accumulate += ' '
        else:
            accumulate += 'y'

        if not accumulate:
            return result

        accumulate6 = ''
        accumulate5 = ''
        accumulate4 = ''
        accumulate3 = ''

        accumulate_length = len(accumulate)
        if accumulate_length == 7:
            accumulate6 = accumulate[1:7]
            accumulate5 = accumulate[2:7]
            accumulate4 = accumulate[3:7]
            accumulate3 = accumulate[4:7]

        elif accumulate_length == 6:
            accumulate6 = accumulate
            accumulate5 = accumulate[1:6]
            accumulate4 = accumulate[2:6]
            accumulate3 = accumulate[3:6]

        elif accumulate_length == 5:
            accumulate5 = accumulate
            accumulate4 = accumulate[1:5]
            accumulate3 = accumulate[2:5]

        elif accumulate_length == 4:
            accumulate4 = accumulate
            accumulate3 = accumulate[1:4]

        elif accumulate_length == 3:
            accumulate3 = accumulate

        if accumulate_length == 7:
            if accumulate == ' xxx x ' or accumulate == ' x xxx ':
                result['open_three'] -= 1

            accumulate = accumulate6

        if len(accumulate6) == 6:
            if accumulate6 == 'xx xxx':
                result['four'] -= 1

            if accumulate6 == ' xx x ' or accumulate6 == ' x xx ':
                result['open_three'] += 1
                result['two'] -= 2

            if accumulate6 == ' xxxx ':
                result['open_four'] += 1
                result['four'] -= 2

            if accumulate6 == 'x xxx ' or accumulate6 == ' xxx x':
                result['three'] += 1

        if len(accumulate5) == 5:
            if accumulate5 == ' xxx ' or accumulate5 == ' xxx ':
                result['open_three'] += 1
                result['three'] -= 2

            if accumulate5 == 'xx xx':
                result['four'] += 1
                result['two'] -= 2

            if accumulate5 == ' xxxx' or accumulate5 == 'xxxx ' or accumulate5 == 'xxx x' or accumulate5 == 'x xxx':
                result['four'] += 1
                result['three'] -= 1

            if accumulate5 == 'xxxxx':
                result['five'] += 1
                self.finish = True
                self.winner = turn
                return accumulate

        if len(accumulate4) == 4:
            if accumulate4 == 'xxx ' or accumulate4 == ' xxx':
                result['three'] += 1
                result['two'] -= 1

        if len(accumulate3) == 3:
            if accumulate3 == 'xx ' or accumulate3 == ' xx':
                result['two'] += 1

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

    def __str__(self):
        to_return = ''
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i][j]
                to_return = str(value)
                if value == 1:
                    to_return += str(value)
                if value == 2:
                    to_return += str(value)

                to_return += ' '

            to_return += '\n'

        return to_return

    def valid_move(self, move):
        i = move / 15
        j = move % 15
        return self.board[i][j] == 0

    def done(self):
        return self.finish