from typing import Tuple, List
from agents.agent import Agent
import torch
from torch.autograd import Variable
import torch.nn.functional as F

import copy
import random
import logging
import numpy as np
from .dqn_net import DQNNet

from game.state import State
from .memory import Memory


class DQNAgent(Agent):
    def __init__(self, board_size: int) -> None:
        Agent.__init__(self, board_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DQNNet().to(self.device)
        self.duplicate_model = copy.deepcopy(self.model).to(self.device)

        self.memory = Memory()
        self.gamma = 0.5

        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01

        self.last_action = None
        self.last_reward = None
        self.last_state = None
        
        self.steps = 0
        self.target_update_freq = 10

    def prepare_for_new_game(self) -> None:
        self.last_state = None
        self.last_action = None

    def act(self, observation: State, reward: float, done: bool) -> int:
        if done:
            # If done is true that means we lose the game
            self.remember(-1.0, observation, done)

            # Whatever we return here is not important anymore
            return -1
        else:
            self.remember(reward, observation, done)

            _, best_move = self.find_best_move(observation)

            self.last_state = observation
            self.last_action = best_move

            return best_move

    def remember(self, reward: float, next_state: State, done: bool) -> None:
        if self.last_state is not None and self.last_action is not None:
            self.augment_and_queue(self.last_state, self.last_action, reward, next_state, done)
            self.prepare_for_new_game()

    def augment_and_queue(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        board = np.array(state.board)
        next_board = np.array(next_state.board)
        idx_matrix = np.arange(self.board_size * self.board_size).reshape((self.board_size, self.board_size))
        
        for k in range(4):
            for flip in [False, True]:
                b = board.copy()
                nb = next_board.copy()
                idx_m = idx_matrix.copy()
                
                if flip:
                    b = np.fliplr(b)
                    nb = np.fliplr(nb)
                    idx_m = np.fliplr(idx_m)
                    
                b = np.rot90(b, k=k)
                nb = np.rot90(nb, k=k)
                idx_m = np.rot90(idx_m, k=k)
                
                new_r, new_c = np.where(idx_m == action)
                new_action = int(new_r[0] * self.board_size + new_c[0])
                
                aug_state = State(self.board_size, b.tolist(), state.turn)
                aug_next_state = State(self.board_size, nb.tolist(), next_state.turn)
                
                self.memory.queue((aug_state, new_action, reward, aug_next_state, done))

    def find_best_move(self, state: State, model: DQNNet=None) -> Tuple[float, int]:
        """
        Find the best legal move given a state of the game
        """
        if random.uniform(0, 1) < self.epsilon:
            return self.find_random_move(state)

        if model is None:
            model = self.model

        with torch.no_grad():
            act_value = model.predict(state, self.device).squeeze(0)
            
        valid_moves = [i for i in range(225) if state.valid_move(i)]
        if not valid_moves:
            return 0.0, 0
            
        mask = torch.ones(225, dtype=torch.bool, device=self.device)
        mask[valid_moves] = False
        act_value[mask] = -float('inf')
        
        max_value, max_value_idx = torch.max(act_value, 0)
        return max_value.item(), max_value_idx.item()

    def find_random_move(self, state: State) -> Tuple[float, int]:
        valid_moves = [i for i in range(225) if state.valid_move(i)]
        if not valid_moves:
            return 0.0, 0
        random_move = random.choice(valid_moves)
        return 0.0, random_move

    def find_best_moves_batch(self, states: List[State]) -> List[int]:
        state_tensor = torch.cat([DQNNet.state_to_tensor(s) for s in states]).to(self.device)
        
        with torch.no_grad():
            act_values = self.model(state_tensor).cpu().numpy()
            
        actions = []
        for i, state in enumerate(states):
            if np.random.rand() <= self.epsilon:
                actions.append(self.find_random_move(state)[1])
            else:
                q_vals = act_values[i]
                valid_moves = [m for m in range(225) if state.valid_move(m)]
                if not valid_moves:
                    actions.append(-1)
                    continue
                    
                masked_q = np.full(225, -np.inf)
                for m in valid_moves:
                    masked_q[m] = q_vals[m]
                    
                actions.append(int(np.argmax(masked_q)))
                
        return actions

    def replay(self, batch_size: int) -> None:
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)

        states = torch.cat([DQNNet.state_to_tensor(s) for s, a, r, ns, d in batch]).to(self.device)
        actions = torch.LongTensor([[a] for s, a, r, ns, d in batch]).to(self.device)
        rewards = torch.FloatTensor([r for s, a, r, ns, d in batch]).to(self.device)
        next_states = torch.cat([DQNNet.state_to_tensor(ns) for s, a, r, ns, d in batch]).to(self.device)
        dones = torch.FloatTensor([1.0 if d else 0.0 for s, a, r, ns, d in batch]).to(self.device)

        # Current Q values
        q_values = self.model(states)
        q_values_for_actions = q_values.gather(1, actions).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q_values = self.duplicate_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = F.mse_loss(q_values_for_actions, targets)

        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        print(f'Batch average loss: {loss.item():.8f}')

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.duplicate_model.load_state_dict(self.model.state_dict())

    @staticmethod
    def log_batch(state: State, action: int, reward: float, next_state: State) -> None:
        logging.debug('State: ' + '\n' + str(state))
        logging.debug('Action: ' + str(action))
        logging.debug('Reward: ' + str(reward))
        logging.debug('Next state: ' + '\n' + str(next_state) + '\n')

    def print_memory(self) -> None:
        """
        Print memory of this agent, for debugging purpose only
        """
        print('Memory size {}'.format(len(self.memory)))
        print(self.memory)

    def save(self, name: str) -> None:
        torch.save(self.model.state_dict(), name)

    def load(self, name: str) -> None:
        self.model.load_state_dict(torch.load(name))
        self.duplicate_model = copy.deepcopy(self.model)
