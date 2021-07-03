from network import Model
from create_data_matrices import get_data
import torch.nn as nn

input, output = get_data()
model = Model(input_dims=input[1].shape, n_actions=5)

EPOCHS = 300

for i in range(EPOCHS):
    features = input[i]
    labels = output[i]

    # run through model
    prediction = model.forward(features)

    # loss
    loss = nn.MSELoss()


