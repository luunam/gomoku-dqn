from game.state import State


def convert_file_to_state(name):
    file_name = name
    with open(file_name) as f:
        content = f.readlines()

        size = int(content[0])

        board = [list(content[i].rstrip()) for i in range(1, size + 1)]

        for i in range(0, size):
            for j in range(0, size):
                if board[i][j] == '~' or board[i][j] == '.':
                    board[i][j] = 0
                elif board[i][j] == 'X':
                    board[i][j] = 1
                elif board[i][j] == 'O':
                    board[i][j] = 2

        state = State(size, board=board)

        return state
