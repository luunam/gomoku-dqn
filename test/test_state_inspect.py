import unittest

from utils.state_loader import *


class TestStateInspect(unittest.TestCase):
    def test_board1(self):
        state = convert_file_to_state('./test/resources/board1')
        result_x = state.inspect(1)
        result_o = state.inspect(2)

        self.assertEqual(result_x['open_three'], 1)
        self.assertEqual(result_x['three'], 1)

        self.assertEqual(result_o['open_three'], 0)
        self.assertEqual(result_o['four'], 1)
        self.assertEqual(result_o['three'], 1)

        self.assertEqual(result_x['five'], 0)
        self.assertEqual(result_o['five'], 0)

    def test_board_2(self):
        state = convert_file_to_state('./test/resources/board2')
        result_x = state.inspect(1)
        result_o = state.inspect(2)

        self.assertEqual(result_x['open_four'], 1)
        self.assertEqual(result_x['open_three'], 1)
        self.assertEqual(result_x['four'], 0)
        self.assertEqual(result_x['three'], 0)

        self.assertEqual(result_o['three'], 0)
        self.assertEqual(result_o['open_three'], 0)
        self.assertEqual(result_o['open_four'], 0)
        self.assertEqual(result_o['four'], 2)

    def test_board3(self):
        state = convert_file_to_state('./test/resources/board3')
        result_x = state.inspect(1)
        result_o = state.inspect(2)

        self.assertEqual(result_x['four'], 0)
        self.assertEqual(result_x['open_four'], 1)
        self.assertEqual(result_x['three'], 0)
        self.assertEqual(result_x['open_three'], 2)

        self.assertEqual(result_o['four'], 5)

        self.assertEqual(result_o['open_four'], 1)
        self.assertEqual(result_o['open_three'], 4)
        self.assertEqual(result_o['three'], 0)

    def test_board5(self):
        state = convert_file_to_state('./test/resources/board5')
        result_x = state.inspect(1)
        result_o = state.inspect(2)

        self.assertEqual(result_x['four'], 1)
        self.assertEqual(result_x['open_four'], 0)
        self.assertEqual(result_x['open_three'], 0)
        self.assertEqual(result_x['three'], 0)

    def test_board8(self):
        state = convert_file_to_state('./test/resources/board8')
        result_x = state.inspect(1)
        result_o = state.inspect(2)

        self.assertEqual(result_x['three'], 4)
        self.assertEqual(result_x['four'], 1)
        self.assertEqual(result_x['open_three'], 1)
        self.assertEqual(result_x['open_four'], 0)

        self.assertEqual(result_o['three'], 0)
        self.assertEqual(result_o['four'], 0)
        self.assertEqual(result_o['open_three'], 0)
        self.assertEqual(result_o['open_four'], 0)

    def test_board10(self):
        state = convert_file_to_state('./test/resources/board10')
        result_x = state.inspect(1)
        result_o = state.inspect(2)

        self.assertEqual(result_o['open_four'], 1)
        self.assertEqual(result_o['three'], 0)
        self.assertEqual(result_o['four'], 0)
        self.assertEqual(result_o['open_three'], 0)

        self.assertEqual(result_x['three'], 1)
        self.assertEqual(result_x['four'], 0)
        self.assertEqual(result_x['open_three'], 0)
        self.assertEqual(result_x['open_four'], 0)

    def test_board15(self):
        state = convert_file_to_state('./test/resources/board15')
        result_x = state.inspect(1)
        result_o = state.inspect(2)

        self.assertEqual(result_x['three'], 2)
        self.assertEqual(result_x['four'], 0)
        self.assertEqual(result_x['open_three'], 0)
        self.assertEqual(result_x['open_four'], 0)

        self.assertEqual(result_o['open_four'], 0)
        self.assertEqual(result_o['three'], 0)
        self.assertEqual(result_o['four'], 0)
        self.assertEqual(result_o['open_three'], 1)


if __name__ == '__main__':
    unittest.main()