import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
# import tensorflow_transform as tft
from tensorflow import keras
from tensorflow.keras import layers
import os


def multi_layer_Perceptron(input_shape=(4, ), hidden_layer_sizes=(2, )):
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    for i, h in enumerate(hidden_layer_sizes):
        model.add(layers.Dense(h, activation='sigmoid', name='hidden_layer_'+str(i)))
    model.add(layers.Dense(3, activation='softmax', name='output_layer'))
    model.summary()

    return model

def plot_history(hist):
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(hist['loss'], lw=3)
    ax.plot(hist['val_loss'], lw=3)
    plt.legend(['Train loss', 'Validation loss'], fontsize=15)
    # ax.set_title('Training loss', size=15)
    ax.set_xlabel('Epochs', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(hist['accuracy'], lw=3)
    ax.plot(hist['val_accuracy'], lw=3)
    plt.legend(['Train Acc.', 'Validation Acc.'], fontsize=15)
    # ax.set_title('Training accuracy', size=15)
    ax.set_xlabel('Epochs', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.show()

# def standardize(inputs):
#     import copy
#     result = copy.copy(inputs)
#     result['features'] = tft.scale_to_z_score(inputs['features'])
#     return result

iris, iris_info = tfds.load('iris', with_info=True)
# print(iris_info)

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

# ds_train_orig['features'] = tft.scale_to_z_score(ds_train_orig['features'])
# ds_valid['features'] = tft.scale_to_z_score(ds_valid['features'])
# ds_test['features'] = tft.scale_to_z_score(ds_valid['features'])


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


dir = "iris-classifiers"
if __name__ == '__main__':

    hidden_layer_sizes = ((2,), (5,), (10,), (5, 5), (10, 5), (10, 10), (20,20))
    for hls in hidden_layer_sizes:
        callbacks = []
        filename = str(hls) + '_hidden_layer_neurons.h5'
        save_best_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(dir, filename), monitor='val_loss',
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
        plot_history(hist)

        # save history
        filename = str(hls) + 'history.npy'
        np.save(os.path.join(dir, filename), hist)

        ## save model and weights
        # filename = str(hls)+'_hidden_layer_neurons.h5'
        # model.save(os.path.join(dir, filename), overwrite=True, include_optimizer=True, save_format='h5')

