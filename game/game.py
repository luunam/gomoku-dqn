from state import State
from colorama import init

class Game:
    def __init__(self, agent1, agent2, size=15):
        self.agents = ['ignore', agent1, agent2]
        self.finish = False
        self.ip_address = '127.0.0.1'
        self.turn = 1
        self.board_size = size
        self.previous_state = State(self.board_size)
        self.state = State(self.board_size)
        self.last_action = None

        init()

    def _start_game_server(self):
        pass

    def run(self):
        while not self.finish:
            action = self.agents[self.turn].act(self.state, self.last_action)

            if action[0] == -1:
                print 'Game is Draw'
                self.finish = True
                break

            next_state = self.state.next_state(action, self.turn)

            opponent_turn = 3 - self.turn
            reward = next_state.rewards[self.turn]
            if reward > 1:
                print 'REWARD: ' + str(reward)
            opponent_reward = next_state.rewards[opponent_turn]

            opponent_reward = opponent_reward * (1.0 - reward)

            # print 'reward: ' + str(reward)
            if self.last_action is not None:
                self.agents[opponent_turn].remember(self.previous_state, self.last_action, opponent_reward, next_state)

            self.last_action = action
            self.previous_state = self.state
            self.state = next_state
            self.turn = opponent_turn

            self.finish = self.state.done()

            # self.state.print_state()
            # print('')
        print('')
        self.state.print_state()