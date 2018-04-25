from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch
from game import State
import numpy as np


class DQNNet(nn.Module):
    def __init__(self):
        super(DQNNet, self).__init__()
        # The first layer will apply this convolution layer and then use max_pool2d, which will cause the image
        # size reduce to (15 - 2 + 1) / 2 = 7
        self.conv1 = nn.Conv2d(1, 10, kernel_size=2)

        # Size image is now (7 - 2 + 1) = 6
        self.conv2 = nn.Conv2d(10, 20, kernel_size=2)

        # 800 because it is equal to 10 * 10 * 8
        self.fc1 = nn.Linear(720, 400)
        self.fc2 = nn.Linear(400, 225)

        self.conv2_drop = nn.Dropout2d()

        self.optimizer = optim.RMSprop(self.parameters(), lr=0.01)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(self.conv2_drop(self.conv2(x)))

        x = x.view(-1, 720)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def fit_single_data(self, data: torch.FloatTensor, target: torch.FloatTensor) -> float:
        """
        Make a single pass through neural network with 1 data and 1 target

        :param data:
        :param target:
        :return:
        """
        data, target = Variable(data).type(torch.FloatTensor), Variable(target)
        self.optimizer.zero_grad()
        y = self(data)

        loss = F.nll_loss(y, target)
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def fit(self, data_loader: DataLoader, epoch=50):
        """
        Do a full forward and backward path for this neural network

        :param data_loader:
        :param epoch:

        :return:
        """
        self.train()

        for i in range(1, epoch):
            epoch_loss = 0
            for batch_id, (data, target) in enumerate(data_loader):
                epoch_loss += self.fit_single_variable(data, target)

            avg_loss = epoch_loss / len(data_loader.dataset)

    def predict(self, state: State) -> Variable:
        """
        Predict the value of the given state based on

        :param state:

        :return: A Variable of LongTensor with size 1 * 225, containing values for each move,
                 with all illegal moves have value 0
        """
        state_var = state.get_pytorch_variable()
        # assume that the board is 15 * 15
        # TODO: change 15 to board size here

        state_var_reshape = state_var.view(1, 1, 15, 15)
        act_value = self(state_var_reshape)  # type: Variable

        # Set every illegal moves to have value 0
        act_value[state_var > 0] = 0

        return act_value