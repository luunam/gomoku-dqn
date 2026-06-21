import sys
import numpy as np
from colorama import Fore, Style
import copy
import torch
from torch.autograd import Variable
from typing import Tuple, List, Dict


class State:
    def __init__(self, size, board=None, turn=2, win_length=5):
        self.size = size
        self.win_length = win_length
        self.action_size = self.size * self.size
        if board is None:
            self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        else:
            self.board = board

        self.finish = False
        self.winner = 0
        self.rewards = ['ignore', 0, 0]
        self.occupied = 0

        # Player's turn that does the last move
        self.turn = turn
        self.win = False

        self.boundary = {
            'maxX': self.size - 1,
            'maxY': self.size - 1,
            'minX': 0,
            'minY': 0
        }

        self.last_action = None

    def get_score(self) -> int:
        return self.rewards[3-self.turn] - self.rewards[self.turn]

    def check_win(self, board: List[List[int]], action: int, turn: int) -> bool:
        x, y = self.convert_to_move(action)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            for step in range(1, self.win_length):
                nx, ny = x + step*dx, y + step*dy
                if 0 <= nx < self.size and 0 <= ny < self.size and board[nx][ny] == turn:
                    count += 1
                else:
                    break
            for step in range(1, self.win_length):
                nx, ny = x - step*dx, y - step*dy
                if 0 <= nx < self.size and 0 <= ny < self.size and board[nx][ny] == turn:
                    count += 1
                else:
                    break
            if count >= self.win_length:
                return True
        return False

    def step(self, action: int) -> Tuple['State', float, bool]:
        """
        Return the next state and reward given the action and the turn of the player that just move. The reward here is the
        reward for the NEXT player, and not the player that just executes the move
        """
        current_turn = self.turn
        next_turn = 3 - self.turn

        clone_board = self._clone_board()

        x, y = self.convert_to_move(action)
        if self.board[x][y] != 0:
            return State(self.size, clone_board, self.turn, self.win_length), -1, True

        clone_board[x][y] = current_turn

        next_state = State(self.size, clone_board, next_turn, self.win_length)
        next_state.occupied = self.occupied + 1
        
        is_win = self.check_win(clone_board, action, current_turn)
        done = is_win or next_state.occupied == self.size * self.size

        # reward logic is handled outside. 0.0 for step
        reward = 0.0

        if done:
            next_state.finish = True
            if is_win:
                next_state.winner = current_turn
            
        next_state.last_action = action
        next_state.boundary = self._update_boundary(action)

        return next_state, reward, done

    def reflected_state(self) -> 'State':
        clone_board = self._clone_board()
        for i in range(self.size):
            for j in range(self.size):
                if clone_board[i][j] == 1:
                    clone_board[i][j] = 2
                elif clone_board[i][j] == 2:
                    clone_board[i][j] = 1

        return State(self.size, clone_board, win_length=self.win_length)

    def get_reward(self, turn: int) -> Tuple[float, bool]:
        """
        Get reward for player with the given turn
        :param turn: the turn of the given player
        :return: reward and whether the game is done
        """
        result = self.inspect(turn)

        done = result['five'] > 0 or self.occupied == self.size * self.size
        return 0.0, done

    def inspect(self, turn: int) -> dict:
        """
        Inspect the current board, return the information on how many connected strings are made
        :param turn:
        :return:
        """
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

        # Traverse sum diagonal (diagonal whose i+j are constant)
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

        # Traverse diff diagonal (diagonal whose i+j are constant):
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

    def evaluate(self, i: int, j: int, accumulate: str, turn: int, result: dict) -> str:
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

    def print_state(self):
        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i][j]
                if value == 1:
                    to_print = Fore.GREEN + str(value) + Style.RESET_ALL
                elif value == 2:
                    to_print = Fore.RED + str(value) + Style.RESET_ALL
                else:
                    to_print = '.'

                sys.stdout.write(to_print + ' ')

            sys.stdout.write('\n')

    def __str__(self):
        to_return = ''
        for i in range (self.size + 1):
            if i >= 1:
                to_return += str((i-1) % 10)
            else:
                to_return += ' '
            to_return += ' '

        to_return += '\n'

        for i in range(self.size):
            for j in range(self.size + 1):
                if j == 0:
                    to_return += str(i % 10)

                else:
                    value = self.board[i][j-1]

                    if value == 1:
                        to_return += Fore.GREEN + str(value) + Style.RESET_ALL
                    elif value == 2:
                        to_return += Fore.RED + str(value) + Style.RESET_ALL
                    else:
                        to_return += '.'

                to_return += ' '

            to_return += '\n'

        return to_return

    def possible_next_states(self, use_boundary: bool = False) -> List['State']:
        opponent_turn = 3 - self.turn
        possible_states = []

        max_i = self.size if not use_boundary else min(self.size, self.boundary['maxX'] + 2)
        max_j = self.size if not use_boundary else min(self.size, self.boundary['maxY'] + 2)
        min_i = 0 if not use_boundary else max(0, self.boundary['minX'] - 1)
        min_j = 0 if not use_boundary else max(0, self.boundary['minY'] - 1)

        for i in range(min_i, max_i):
            for j in range(min_j, max_j):
                if self.board[i][j] == 0:
                    new_state = self.next_state(i * self.size + j, opponent_turn)
                    possible_states.append(new_state)

        return possible_states

    def valid_move(self, move: int) -> bool:
        i = move // self.size
        j = move % self.size
        return self.board[i][j] == 0

    def done(self) -> bool:
        return self.finish

    def convert_to_move(self, n: int) -> Tuple[int, int]:
        return n // self.size, n % self.size

    def _update_boundary(self, action: int) -> dict:
        x = action // self.size
        y = action % self.size
        return {
            'maxX': max(self.boundary['maxX'], x),
            'minX': min(self.boundary['minX'], x),
            'maxY': max(self.boundary['maxY'], y),
            'minY': min(self.boundary['minY'], y)
        }

    def _clone_board(self) -> List[List[int]]:
        return [row[:] for row in self.board]
