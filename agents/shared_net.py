import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from game.state import State
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class SharedNet(nn.Module):
    def __init__(self, size=15) -> None:
        super(SharedNet, self).__init__()
        self.size = size
        
        # Initial convolutional block
        self.conv_in = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(64)
        
        # 5 Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock(64) for _ in range(5)]
        )
        
        # Policy Head (Actor)
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * size * size, size * size)
        
        # Value Head (Critic)
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * size * size, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(-1, 2 * self.size * self.size)
        policy_logits = self.policy_fc(p)
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(-1, 1 * self.size * self.size)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy_logits, value

    @staticmethod
    def state_to_tensor(state: State) -> torch.Tensor:
        """
        Convert state to pytorch tensor using a relative perspective
        """
        board = np.asarray(state.board)
        my_piece = 3 - state.turn
        opp_piece = state.turn
        
        rel_board = np.zeros_like(board, dtype=np.float32)
        rel_board[board == my_piece] = 1.0
        rel_board[board == opp_piece] = -1.0
        
        state_tensor = torch.from_numpy(rel_board)
        state_tensor = state_tensor.view(1, 1, state.size, state.size)
        return state_tensor
