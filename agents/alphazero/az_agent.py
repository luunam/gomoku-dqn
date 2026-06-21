import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from agents.agent import Agent
from agents.shared_net import SharedNet
from .mcts import MCTS
from game.state import State
import random

class AZMemory:
    def __init__(self):
        self.states = []
        self.mcts_probs = []
        self.winners = []
        
    def clear(self):
        self.states.clear()
        self.mcts_probs.clear()
        self.winners.clear()

class AlphaZeroAgent(Agent):
    def __init__(self, board_size=15):
        super().__init__(board_size)
        self.board_size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SharedNet(board_size).to(self.device)
        self.memory = AZMemory()
        
        self.mcts = MCTS(self.model, num_simulations=50, device=self.device)
        self.batch_size = 256
        self.epsilon = 0.0 # for compatibility with evaluate_against_random

    def act(self, state: State, reward: float = 0, done: bool = False):
        return self.find_best_move(state)

    def find_best_move(self, state: State):
        probs = self.mcts.search(state)
        action = np.argmax(probs)
        return int(action)

    def find_best_moves_batch(self, states: List[State]) -> Tuple[List[int], List[np.ndarray]]:
        actions = []
        all_probs = []
        
        for state in states:
            probs = self.mcts.search(state)
            
            # Dirichlet noise for exploration
            noise = np.random.dirichlet([0.03] * len(probs))
            probs = 0.75 * probs + 0.25 * noise
            probs = probs / np.sum(probs)
            
            action = np.random.choice(len(probs), p=probs)
            actions.append(int(action))
            all_probs.append(probs)
            
        return actions, all_probs

    def replay(self, batch_size: int = 0) -> None:
        if len(self.memory.states) < self.batch_size:
            return
            
        states = torch.cat(self.memory.states).to(self.device)
        mcts_probs = torch.tensor(np.array(self.memory.mcts_probs), dtype=torch.float32).to(self.device)
        winners = torch.tensor(self.memory.winners, dtype=torch.float32).unsqueeze(-1).to(self.device)
        
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        
        for start in range(0, dataset_size, self.batch_size):
            end = start + self.batch_size
            if end > dataset_size:
                break
                
            batch_idx = indices[start:end]
            
            b_states = states[batch_idx]
            b_probs = mcts_probs[batch_idx]
            b_winners = winners[batch_idx]
            
            logits, values = self.model(b_states)
            
            policy_loss = -torch.sum(b_probs * F.log_softmax(logits, dim=-1), dim=-1).mean()
            value_loss = F.mse_loss(values, b_winners)
            loss = policy_loss + value_loss
            
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()
            
        self.memory.clear()

    def save(self, file_name: str) -> None:
        torch.save(self.model.state_dict(), file_name)

    def load(self, file_name: str) -> None:
        self.model.load_state_dict(torch.load(file_name, map_location=self.device))
        self.model.eval()
        self.model.train()
        
    def remember(self, reward, state, done):
        pass
