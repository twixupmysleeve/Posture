from network import Model
from Posture.data_processing.create_data_matrices import get_data
import torch as T
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn

input, output = get_data()
print(input)
print(output)
model = Model(input_dims=5, n_actions=5, lr=1)

EPOCHS = 300
losses = []

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay= 1e-6)

for epoch in range(EPOCHS):

    epoch_loss = 0

    for i in range(len(input)):

        features = T.tensor(input[i], dtype=T.float32)
        labels = T.tensor(output[i], dtype=T.float32)

        model.optim.zero_grad()

        # forward prop
        prediction = model.forward(features)

        # loss
        loss = loss_function(prediction, labels)

        # backprop
        loss.backward()

        epoch_loss += loss.detach().numpy()

        model.optim.step()

    losses.append(epoch_loss)

    print(f"epoch {epoch} loss {epoch_loss}")

plt.plot(losses)
plt.show()
