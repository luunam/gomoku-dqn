from collections import deque
import copy
import random


class Memory:
    def __init__(self, max_length=5000):
        self.ephemeral_storage = deque()
        self.permanent_storage = deque()
        self.max_length = max_length
        self.stack_size = 1

    def queue(self, memory):
        _, _, _, _, done = memory

        self.ephemeral_storage.append(memory)

        if done:
            self.stack_memory()

    def queue_permanent_storage(self, memory):
        if len(self.permanent_storage) >= self.max_length:
            self.permanent_storage.popleft()

        self.permanent_storage.append(memory)

    def stack_memory(self):
        ephemeral_len = len(self.ephemeral_storage)
        for idx, val in enumerate(self.ephemeral_storage):
            if idx < ephemeral_len - self.stack_size:
                merged_mem = self.merge(val, self.ephemeral_storage[idx+self.stack_size])
                self.queue_permanent_storage(merged_mem)

        self.ephemeral_storage.clear()

    def merge(self, mem1, mem2):
        state1, action1, reward1, next_state1, done1 = mem1
        state2, action2, reward2, next_state2, done2 = mem2

        return state1, action1, reward1 + reward2, next_state2, done2

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