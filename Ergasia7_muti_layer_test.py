import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from Ergasia7_multi_layer_train import ds_test
from Ergasia7_multi_layer_train import plot_history

dir = "iris-classifiers"
hidden_layer_sizes = ((2,), (5,), (10,), (5, 5), (10, 5), (10, 10), (20,20))
for hls in hidden_layer_sizes:
    filename = str(hls) + '_hidden_layer_neurons.h5'
    model = tf.keras.models.load_model(os.path.join(dir, filename))
    results = model.evaluate(ds_test.batch(30), verbose=0)

    print('Test metrics for model with hidden layer sizes ' + str(hls) + ':')
    print('Test loss: {:.4f} Test Acc.: {:.4f}\n'.format(*results))


    filename = str(hls) + 'history.npy'
    hist = np.load(os.path.join(dir, filename), allow_pickle=True).item()
    plot_history(hist)

