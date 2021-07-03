import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np


def confusion(label, pred, confidence=0.7):
    n_dims = pred.shape[1]
    N = pred.shape[0]
    confus = np.zeros(shape=(N, N))

    for m in range(n_dims):
        for n in N:
            n_pred = pred[m][n]
            actual = label[m][n]

            pred_val = 1 if n_pred>=confidence else 0

            if pred_val==actual:
                confus[n][n]+=1


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
