from collections import deque
import copy
import random
import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedMemory:
    def __init__(self, max_length=200000, e=0.01, a=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree = SumTree(max_length)
        self.e = e
        self.a = a
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def queue(self, memory):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = 1.0
        self.tree.add(max_p, memory)

    def sample(self, sample_size=32):
        batch = []
        idxs = []
        segment = self.tree.total() / sample_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(sample_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = self._get_priority(error)
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries

    def __str__(self):
        return f"PrioritizedMemory with {self.tree.n_entries} entries"


class Memory:
    def __init__(self, max_length=200000):
        self.permanent_storage = deque()
        self.max_length = max_length

    def queue(self, memory):
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