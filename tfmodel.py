import tensorflow as tf
from create_data_matrices import get_data
import matplotlib.pyplot as plt
import numpy as np

# TODO: add testing data
# TODO: randomize data

input, output = get_data()

print(input.shape)
print(output.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, 5)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(5),
    tf.keras.layers.Flatten(input_shape=(1, 5)),
])


opt = tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'],
              optimizer=opt)

hist = model.fit(input, output, epochs=300)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

acc = hist.history['accuracy']
loss = hist.history['loss']

# epochs = range(1, len(acc) + 1)

plt.plot(running_mean(acc, 1), label='Training acc')
plt.title('Training Accuracy')
plt.legend()

plt.figure()

plt.plot(running_mean(loss, 1), label='Training loss')
plt.title('Training Loss')
plt.legend()

plt.show()


