from collections import deque
import copy
import random


class Memory:
    def __init__(self, max_length=200000):
        self.permanent_storage = deque()
        self.max_length = max_length

    def queue(self, memory):
        """Add a transition directly to permanent storage."""
        if len(self.permanent_storage) >= self.max_length:
            self.permanent_storage.popleft()

        self.permanent_storage.append(memory)

    def sample(self, sample_size=32):
        if sample_size > len(self.permanent_storage):
            return copy.deepcopy(self.permanent_storage)

        return random.sample(self.permanent_storage, sample_size)

    def __len__(self):
        return len(self.permanent_storage)

    def __str__(self):
        to_return = ''
        for state, action, reward, next_state, done in self.permanent_storage:
            to_return += 'State: ' + '\n' + str(state) + '\n'
            x = action // 15
            y = action % 15
            to_return += 'Action: ' + '\n' + str(x) + ' ' + str(y) + '\n'
            to_return += 'Reward: ' + '\n' + str(reward) + '\n'
            to_return += 'Next state: ' + '\n' + str(next_state) + '\n'
            to_return += 'Done: ' + '\n' + str(done) + '\n'
        return to_return