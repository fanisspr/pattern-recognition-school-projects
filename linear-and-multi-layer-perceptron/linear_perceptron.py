'''
The Iris data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). 
Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
Iris Setosa is the only one that is linearly seperable from the others

In this project, a linear perceptron is used to classify samples from the iris dataset
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron


X, y = load_iris(return_X_y=True)
# Split data into train, test
X_train = np.concatenate([X[i:(i+40), :] for i in [0,50,100]])
X_test = np.concatenate([X[(i+40):(i+50), :] for i in [0,50,100]])

y_train = np.concatenate([y[i:(i + 40)] for i in [0, 50, 100]])
y_test = np.concatenate([y[(i+40):(i+50)] for i in [0,50,100]])

# standardize train/test dataset
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# shuffle train dataset
np.random.seed(1)
p = np.random.permutation(X_train.shape[0])
X_train = X_train[p]
y_train = y_train[p]

num_epochs = 100
lr = 0.01

lin_percep = Perceptron(random_state=0, 
                        max_iter=num_epochs, 
                        eta0=lr, 
                        early_stopping=True, 
                        validation_fraction=0.3, 
                        n_iter_no_change=5, 
                        verbose=1)
lin_percep.fit(X_train, y_train)

print("\nLinear Perceptron accuracy:")

y_pred = lin_percep.predict(X_train)
accuracy = np.mean(y_train==y_pred)
print("Train accuracy: {0:.4f}".format(accuracy))

y_pred = lin_percep.predict(X_test)
accuracy = np.mean(y_test==y_pred)
print("Test accuracy: {0:.4f}".format(accuracy))
