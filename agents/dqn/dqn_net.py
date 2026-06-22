from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch
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

class DQNNet(nn.Module):
    def __init__(self, board_size=15) -> None:
        super(DQNNet, self).__init__()
        self.board_size = board_size
        # Initial convolutional block
        self.conv_in = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(64)
        
        # 5 Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock(64) for _ in range(5)]
        )
        
        # Output head
        self.conv_out = nn.Conv2d(64, 2, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(2)
        
        self.fc = nn.Linear(2 * self.board_size * self.board_size, self.board_size * self.board_size)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial block
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        # ResNet body
        x = self.res_blocks(x)
        
        # Output block
        x = F.relu(self.bn_out(self.conv_out(x)))
        
        x = x.view(-1, 2 * self.board_size * self.board_size)
        x = self.fc(x)
        return x

    @staticmethod
    def state_to_tensor(state: State) -> torch.Tensor:
        """
        Convert state to pytorch tensor using a relative perspective
        """
        board = np.asarray(state.board)
        my_piece = state.turn
        opp_piece = 3 - state.turn
        
        rel_board = np.zeros_like(board, dtype=np.float32)
        rel_board[board == my_piece] = 1.0
        rel_board[board == opp_piece] = -1.0
        
        state_tensor = torch.from_numpy(rel_board)
        state_tensor = state_tensor.view(1, 1, state.size, state.size)
        return state_tensor

    @staticmethod
    def state_to_variable(state: State) -> Variable:
        """
        Convert state to pytorch variable
        """
        state_tensor = DQNNet.state_to_tensor(state)
        state_var = Variable(state_tensor).type(torch.FloatTensor)
        return state_var

    def predict(self, state: State, device: torch.device = None) -> torch.Tensor:
        """
        Predict the Q-values of the given state.
        
        :param state: State
        :return: A Tensor of size 1 * 225 containing values for each move
        """
        state_tensor = DQNNet.state_to_tensor(state)
        if device is not None:
            state_tensor = state_tensor.to(device)
        
        # Set to eval mode to use moving average statistics for BatchNorm
        was_training = self.training
        self.eval()
        
        act_value = self(state_tensor)
        
        if was_training:
            self.train()
            
        return act_value
