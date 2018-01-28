from utils import *


class Benchmarker:
    def __init__(self):
        self.tests = [{'state': None, 'expected_move': None}]
        for i in range(1, 6):
            self.tests.append({
                'state': convert_file_to_state('./benchmark/resources/board' + str(i))
            })

        self.tests[1]['expected_move'] = [(4, 5), (9, 5)]
        self.tests[2]['expected_move'] = [(5, 5), (7, 5)]
        self.tests[3]['expected_move'] = [(4, 6), (8, 10)]
        self.tests[4]['expected_move'] = [(8, 10)]
        self.tests[5]['expected_move'] = [(4, 5)]

    def rate(self, agent):
        score = 0
        for i in range(1, len(self.tests)):
            test = self.tests[i]
            action = agent.act(test['state'])
            x = action / test['state'].size
            y = action % test['state'].size
            print('Test ' + str(i) + ': Agent move ' + str(x) + ', ' + str(y))
            for expected_move in test['expected_move']:
                if self.move_equal(x, y, expected_move[0], expected_move[1]):
                    score += 1
                    print('Pass test ' + str(i))
                    break

        return score

    def move_equal(self, x1, y1, x2, y2):
        return x1 == x2 and y1 == y2