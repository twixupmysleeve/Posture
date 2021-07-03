import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class Model(nn.Module):
    def __init__(self, input_dims=5, n_actions=5, fc1_dims=8, fc2_dims=8, lr=0.003):
        super(Model, self).__init__()

        self.input =  nn.Linear(input_dims, fc1_dims)
        self.fc1 = nn.Linear(fc1_dims, fc2_dims)
        self.output = nn.Linear(fc2_dims, n_actions)

        self.optim = Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()


    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.output(x)
        x = F.relu(x)

        return x
