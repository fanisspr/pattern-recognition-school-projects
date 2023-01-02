import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras import layers


def get_batch(X, Y, batch_size, idx):
    return X[batch_size*idx:batch_size*(idx+1)], Y[batch_size*idx:batch_size*(idx+1)]


def dense(x, W, b):
    return tf.sign(tf.matmul(x, W) + b)

# def step_func(act: tf.Tensor, X):
#     # sign = tf.sign(x)
#     with tf.Session() as sess:
#         act = sess.run(act, feed_dict={x: X})
#     # x = x.numpy()
#         pred = np.array([1.0 if a>=0 else 0.0 for i in act for a in i])
#     # for i in act:
#     #     for j in i:
#     #         if
#     #     pred = tf.Tensor(pred) #tf.convert_to_tensor
#         pred = tf.convert_to_tensor(pred)
#         sess.close()
#     return pred


X, Y = load_iris(return_X_y=True)

norm = MinMaxScaler().fit(X)
X = norm.transform(X) #normalize gives worse accuracy
# X = (X - np.mean(X))/np.std(X)

# Changing labels to one-hot encoded vector
lb = LabelBinarizer()
Y = lb.fit_transform(Y)
# Changing 0 to -1 to fit with sign function
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        if Y[i, j] == 0:
            Y[i, j] = -1

# Split data into train, test
X_train = np.concatenate([X[i:(i+40), :] for i in [0,50,100]])
X_test = np.concatenate([X[(i+40):(i+50), :] for i in [0,50,100]])

Y_train = np.concatenate([Y[i:(i+40)] for i in [0,50,100]])
Y_test = np.concatenate([Y[(i+40):(i+50)] for i in [0,50,100]])

# shuffle data
p = np.random.permutation(X_train.shape[0])
X_train = X_train[p]
Y_train = Y_train[p]
# Y = Y.reshape(-1,1)


# define hyperparameters
num_features = X.shape[1]
num_classes = 3
num_epochs = 50
batch_size = 1
steps_per_epoch = int(X_train.shape[0] / batch_size)
lr = 0.01

# define placeholder for input and output
x = tf.placeholder(tf.float64, [None, num_features])
y = tf.placeholder(tf.float64, [None, num_classes])
# weights and bias
w = tf.Variable(tf.random_normal([num_features, num_classes], dtype=tf.float64))
b = tf.Variable(tf.random_normal([num_classes], dtype=tf.float64))


activations = tf.sign(tf.add(tf.matmul(x, w), b))

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=activations)) #Maybe change to sum-of-sqaures error  #shlould create 1 loss for each neuron? no
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

init = tf.global_variables_initializer()

# MAYBE SHOULD CHANGE STUFF HERE
with tf.Session() as sess:
    sess.run(init)
    # training cycle
    for e in range(num_epochs):
        avg_loss = 0
        # loop over all batches
        for i in range(steps_per_epoch):
            batch_x, batch_y = get_batch(X_train, Y_train, batch_size, i)

            _, c = sess.run([optimizer, loss], feed_dict={x: batch_x, y: batch_y})
            avg_loss += c / steps_per_epoch
        print("Epoch", e+1, "Loss", avg_loss)

    # Test model
    correct_prediction = tf.equal(tf.argmax(activations, 1), tf.argmax(y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Model accuracy:", accuracy.eval({x: X_test, y: Y_test}))

    sess.close()


# model = keras.Sequential()
# model.add(keras.Input(shape=(X.shape[1], )))
# model.add(layers.Dense(3))
# model.summary()
