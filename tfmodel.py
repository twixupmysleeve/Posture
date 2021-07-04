import pprint
import pandas as pd
import tensorflow as tf
from data_processing.create_data_matrices import get_data
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
# from pycm import *
# import seaborn as sns

USE_MODEL = False

input, output = get_data()

split = int(0.8 * len(input))
(train_features, train_labels), (test_features, test_labels) = (input[:split], output[:split]), \
                                                               (input[split:], output[split:])

if not USE_MODEL:

    model = tf.keras.Sequential([
        # tf.keras.layers.Dense(1, activation='relu'),
        # tf.keras.layers.Dense(3, activation='relu'),
        # tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(5),
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'],
                  optimizer=opt)

    hist = model.fit(train_features, train_labels, epochs=200)

    acc = hist.history['accuracy']
    loss = hist.history['loss']

    valid_loss, valid_acc = model.evaluate(test_features, test_labels)

    print(f"Validation Loss: {valid_loss}\nValidation Accuracy: {valid_acc}")

    # matrix = tf.math.confusion_matrix(tf.Tensor(test_labels), tf.Tensor(model.predict(test_features)))
    # print(matrix)

    model.save("working_model_1")

    plt.plot(acc, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.figure()

    plt.plot(loss, label='Training Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.show()
else:
    model = tf.keras.models.load_model("working_model_1")
    preds = model.predict(test_features)

    cm = ConfusionMatrix(actual_vector=test_labels[0], predict_vector=preds[0])
    print(cm.table)
