import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple
from agents.agent import Agent
from agents.shared_net import SharedNet
from game.state import State
import random

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.values.clear()

class PPOAgent(Agent):
    def __init__(self, board_size=15):
        super().__init__(board_size)
        self.board_size = board_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SharedNet(board_size).to(self.device)
        self.memory = PPOMemory()
        
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.entropy_coef = 0.01
        self.K_epochs = 4
        self.batch_size = 256

    def act(self, state: State, reward: float = 0, done: bool = False):
        return self.find_best_move(state)

    def find_best_move(self, state: State):
        state_tensor = SharedNet.state_to_tensor(state).to(self.device)
        with torch.no_grad():
            logits, value = self.model(state_tensor)
            
            valid_moves = [m for m in range(225) if state.valid_move(m)]
            if not valid_moves:
                return 0
                
            mask = torch.full((1, 225), float('-inf')).to(self.device)
            mask[0, valid_moves] = 0
            
            logits = logits + mask
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            
        return action.item()

    def find_best_moves_batch(self, states: List[State]) -> Tuple[List[int], List[float], List[float]]:
        state_tensor = torch.cat([SharedNet.state_to_tensor(s) for s in states]).to(self.device)
        with torch.no_grad():
            logits, values = self.model(state_tensor)
            
            actions = []
            logprobs = []
            
            for i, state in enumerate(states):
                valid_moves = [m for m in range(225) if state.valid_move(m)]
                if not valid_moves:
                    actions.append(-1)
                    logprobs.append(0.0)
                    continue
                    
                mask = torch.full((225,), float('-inf')).to(self.device)
                mask[valid_moves] = 0
                
                masked_logits = logits[i] + mask
                probs = F.softmax(masked_logits, dim=-1)
                dist = Categorical(probs)
                
                action = dist.sample()
                logprob = dist.log_prob(action)
                
                actions.append(action.item())
                logprobs.append(logprob.item())
                
        return actions, logprobs, values.squeeze(-1).tolist()

    def replay(self, batch_size: int = 0) -> None:
        if len(self.memory.states) == 0:
            return
            
        old_states = torch.cat(self.memory.states).to(self.device)
        old_actions = torch.tensor(self.memory.actions, dtype=torch.long).to(self.device)
        old_logprobs = torch.tensor(self.memory.logprobs, dtype=torch.float32).to(self.device)
        old_values = torch.tensor(self.memory.values, dtype=torch.float32).to(self.device)
        rewards = self.memory.rewards
        is_terminals = self.memory.is_terminals
        
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        
        dataset_size = old_states.size(0)
        indices = np.arange(dataset_size)
        
        for _ in range(self.K_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                b_states = old_states[batch_idx]
                b_actions = old_actions[batch_idx]
                b_logprobs = old_logprobs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]
                
                logits, values = self.model(b_states)
                
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)
                
                new_logprobs = dist.log_prob(b_actions)
                dist_entropy = dist.entropy()
                values = values.squeeze(-1)
                
                ratios = torch.exp(new_logprobs - b_logprobs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages
                
                loss = -torch.min(surr1, surr2) + 0.5 * F.mse_loss(values, b_returns) - self.entropy_coef * dist_entropy
                
                self.model.optimizer.zero_grad()
                loss.mean().backward()
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
