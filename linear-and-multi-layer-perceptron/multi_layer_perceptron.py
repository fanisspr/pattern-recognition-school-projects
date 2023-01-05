'''
The Iris data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). 
Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
Iris Setosa is the only one that is linearly seperable from the others

In this project, a multi-layer perceptron is used to classify samples from the iris dataset
'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import os


root_dir = os.path.relpath(os.path.dirname(__file__))
models_dir = os.path.join(root_dir, 'models-and-history')
if not os.path.exists(models_dir):
    os.mkdir(models_dir)
plots_dir = os.path.join(root_dir, 'plots')
if not os.path.exists(plots_dir):    
    os.mkdir(plots_dir)


def multi_layer_Perceptron(input_shape: tuple[int]=(4,), 
                            hidden_layer_sizes: tuple[int]=(2,)) -> keras.Model:
    """
    Create a multi-layer perceptron model.

    Parameters:
    input_shape: tuple of int, the shape of the input layer
    hidden_layer_sizes: tuple of int, the sizes of the hidden layers

    Returns:
    model: a Keras model with the specified architecture
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    for i, h in enumerate(hidden_layer_sizes):
        model.add(layers.Dense(h, activation='sigmoid', name='hidden_layer_'+str(i)))
    model.add(layers.Dense(3, activation='softmax', name='output_layer'))
    model.summary()
    return model

def plot_history(hist: dict[str, list[float]], filepath: str, title: str):
    """
    Plots the history of the training and validation loss and accuracy for a Keras model.

    Parameters:
    hist: A dictionary with keys 'loss', 'val_loss', 'accuracy', and 'val_accuracy' containing the
            training and validation loss and accuracy values for each epoch.
    filepath: A string with the path to save the plot at
    title: A string with the title for the plots
    
    Returns:
    None. The function plots the history using matplotlib.
    """
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(hist['loss'], lw=3)
    ax.plot(hist['val_loss'], lw=3)
    plt.legend(['Train loss', 'Validation loss'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(hist['accuracy'], lw=3)
    ax.plot(hist['val_accuracy'], lw=3)
    plt.legend(['Train Acc.', 'Validation Acc.'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.title(title)
    fig.savefig(filepath)
    plt.show()
    plt.show()

iris, iris_info = tfds.load('iris', with_info=True)

training_size = 90
validation_size = 30

# Dataset creation, split
tf.random.set_seed(1)
ds = iris['train']
ds = ds.shuffle(150, reshuffle_each_iteration=False)
ds_train_orig = ds.take(training_size)
remaining = ds.skip(training_size)
ds_valid = remaining.take(validation_size)
ds_test = remaining.skip(validation_size)

ds_train_orig = ds_train_orig.map(lambda x: (x['features'], x['label']))
ds_valid = ds_valid.map(lambda x: (x['features'], x['label']))
ds_test = ds_test.map(lambda x: (x['features'], x['label']))

# define hyperparameters
num_epochs = 200
batch_size = 10
steps_per_epoch = np.ceil(training_size / batch_size)
validation_steps = np.ceil(validation_size / batch_size)
lr = 0.01

# shuffle and batch train and valid datasets
ds_train = ds_train_orig.shuffle(buffer_size=training_size)
ds_train = ds_train.repeat()
ds_train = ds_train.batch(batch_size=batch_size)
ds_train = ds_train.prefetch(buffer_size=1000)

ds_valid = ds_valid.shuffle(buffer_size=validation_size)
ds_valid = ds_valid.repeat()
ds_valid = ds_valid.batch(batch_size=batch_size)
ds_valid = ds_valid.prefetch(buffer_size=1000)


if __name__ == '__main__':
    # try 1 hidden layer and 2 hidden layers with different neurons
    hidden_layer_sizes = ((2,), (5,), (10,), (5, 5), (10, 5), (10, 10), (20,20))
    for hls in hidden_layer_sizes:
        callbacks = []
        filename = str(hls) + '_hidden_layer_neurons.h5'
        save_best_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(models_dir, filename), monitor='val_loss',
                                                                save_best_only=True)
        callbacks.append(save_best_callback)

        model = multi_layer_Perceptron(hidden_layer_sizes=hls)
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(
            ds_train,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=ds_valid,
            validation_steps=validation_steps,
            verbose=2,
            callbacks=callbacks)

        #metrics
        results = model.evaluate(ds_test.batch(30), verbose=0)
        print('-----------------------------------------------------------\n')
        print('Test metrics for model with hidden layer sizes ' + str(hls) + ':')
        print('Test loss: {:.4f} Test Acc.: {:.4f}\n'.format(*results))
        print('-----------------------------------------------------------')

        #plot metrics
        hist = history.history
        plot_history(hist, 
                     filepath=os.path.join(plots_dir, f'{hls}-model-plot.png'), 
                     title=f'History of model with hidden layer sizes {hls}')

        # save history
        filename = str(hls) + 'history.npy'
        np.save(os.path.join(models_dir, filename), hist)

        ## save model and weights
        # filename = str(hls)+'_hidden_layer_neurons.h5'
        # model.save(os.path.join(dir, filename), overwrite=True, include_optimizer=True, save_format='h5')

