from .state import State
from colorama import init


class Game:
    def __init__(self, agent1, agent2, size=15, test=False):
        self.agents = [agent1, agent2]
        self.finish = False
        self.ip_address = '127.0.0.1'
        self.turn = 1
        self.board_size = size
        self.previous_state = State(self.board_size)
        self.state = State(self.board_size)
        self.last_action = None
        self.test = test

        init()

    def run(self):
        reward = 0
        done = False
        while not done:
            reward, done = self.calling_all_agents(reward, done)

    def calling_all_agents(self, reward, done):
        for agent in self.agents:
            action = agent.act(self.state, reward, done)
            # reward that self.state.step returns is for the next agent and not for the agent that just makes the move
            self.state, reward, done = self.state.step(action)
            print('Done: {}'.format(done))

            print(self.state)
            if done:
                agent.remember(self.state, reward, done)
                return reward, done

        return reward, done


