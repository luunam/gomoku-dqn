from agents.agent import Agent


class MinimaxAgent(Agent):
    def __init__(self, board_size):
        Agent.__init__(self, board_size)
        self.depth = 1
        self.num_worker = 4

    def act(self, state, reward, done):
        res = self.search(0, state, 0, -1000000, 1000000)
        return res.last_action

    def search(self, depth, state, no_min_agent, alpha, beta):
        if depth == self.depth:
            return state
        elif no_min_agent == 0:
            return self.max_value(state, depth, alpha, beta)
        else:
            return self.min_value(state, depth, alpha, beta)

    def max_value(self, state, depth, alpha, beta):
        # print 'Max value'
        v = -1000000

        possible_moves = state.possible_next_states(use_boundary=True)
        ret = None

        for state in possible_moves:
            if state.finish:
                return state

            new_state = self.search(depth, state, 1, alpha, beta)
            # print 'new state score: ' + str(new_state.get_score())
            if new_state.get_score() > v:
                v = new_state.get_score()

                ret = state

                if v > beta:
                    return ret

                alpha = max(alpha, v)

            # print 'v: ' + str(v)

        return ret

    def min_value(self, state, depth, alpha, beta):
        # print 'Min value: '
        v = 1000000
        possible_moves = state.possible_next_states(use_boundary=True)
        ret = None

        for state in possible_moves:
            if state.finish:
                return state

            new_state = self.search(depth + 1, state, 0, alpha, beta)
            # print 'new state score: ' + str(new_state.get_score())
            if new_state.get_score() < v:
                v = new_state.get_score()
                # print 'v: ' + str(v)
                ret = state

                if v < alpha:
                    return ret

                beta = min(beta, v)

        return ret
