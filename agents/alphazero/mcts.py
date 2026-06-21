import math
import numpy as np
import torch
from agents.shared_net import SharedNet

class Node:
    def __init__(self, state, parent=None, action_taken=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        
        self.children = {}
        
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, action_probs):
        self.is_expanded = True
        for action, prob in action_probs.items():
            if action not in self.children:
                next_state, _, _ = self.state.step(action)
                self.children[action] = Node(next_state, parent=self, action_taken=action, prior=prob)

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            u = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            q = child.value()
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            self.parent.backpropagate(-value)

class MCTS:
    def __init__(self, model, num_simulations=50, c_puct=1.0, device='cpu'):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device

    def search(self, initial_state):
        root = Node(initial_state)

        for _ in range(self.num_simulations):
            node = root
            
            while node.is_expanded:
                action, node = node.select_child(self.c_puct)
            
            if node.state.occupied == node.state.size * node.state.size or node.state.win:
                if node.state.win:
                    node.backpropagate(-1.0)
                else:
                    node.backpropagate(0.0)
                continue

            state_tensor = SharedNet.state_to_tensor(node.state).to(self.device)
            with torch.no_grad():
                logits, value = self.model(state_tensor)
                value = value.item()
                
                valid_moves = [m for m in range(node.state.size * node.state.size) if node.state.valid_move(m)]
                if not valid_moves:
                    node.backpropagate(0.0)
                    continue
                    
                mask = torch.full((1, node.state.size * node.state.size), float('-inf')).to(self.device)
                mask[0, valid_moves] = 0
                
                probs = torch.nn.functional.softmax(logits + mask, dim=-1).squeeze(0).cpu().numpy()
                
            action_probs = {m: probs[m] for m in valid_moves}
            node.expand(action_probs)
            
            node.backpropagate(value)

        action_probs = np.zeros(root.state.size * root.state.size)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count / self.num_simulations
            
        return action_probs
