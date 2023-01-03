import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from numpy.linalg import inv

root_dir = os.path.relpath(os.path.dirname(__file__))

#todo add legends
def plot_data(X, y, title=None, xlabel=None, ylabel=None, theta=None):
    """
        plotData: Function to plot data.
        Using this function you are able to plot your training data with/without your linear regression model.

        Parameters:
        -----------
        X: An [m x n] matrix containing the n-features for m-samples.
        y: A 1-d vector containing the true value for m-samples.
        theta: parameters (aka weights)
    """

    # find indices of each class
    pos = y == 1
    neg = y == -1

    plt.figure()
    plt.plot(X[pos, 1], X[pos, 2], 'k*', lw=2, ms=10)
    plt.plot(X[neg, 1], X[neg, 2], 'ko', mfc='y', ms=8, mec='k', mew=1)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if theta is not None:
        plt.plot(X[:, 1], np.matmul(X[:, 0:2], theta[0:2]), color='blue', linewidth=3,
                 label='Linear Regression')  # Line visualization
        # X2 = X[:, 3:]
        # X2 = np.column_stack((np.ones((X.shape[0])), X2))
        # # theta2 = theta[3:4]
        # # theta2 = np.column_stack((theta[0], theta2))
        # plt.plot(X2[:, 1], np.matmul(X2[:, 0:2], [theta[0], theta[3]]), color='blue', linewidth=3,
        #          label='Linear Regression')  # Line visualization

    plt.legend(loc="upper left")
    plt.show()


def feature_standardize(X):
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    for i in range(0, X.shape[1]):
        mu[:, i] = np.mean(X[:, i])
        sigma[:, i] = np.std(X[:, i])
        X_norm[:, i] = (X[:, i] - float(mu[:, i])) / float(sigma[:, i])

    return X_norm, mu, sigma


def predict(X, theta, bias = 0):

    activation = np.dot(X, theta)
    # pred = np.array([1.0 if a >= bias else 0.0 for a in activation])
    pred = np.array([1.0 if a >= bias else -1.0 for a in activation])

    return pred


def batch_perceptron(X, y, alpha, num_iters):
    """
    Estimate perceptron weights using gradient descent with gradient of cost function J: = -Σy
    where y = falsely classified samples

    Parameters:
        -----------
        X: An [m x n] matrix containing the n-features for m-samples.
        y: A 1-d vector containing the true value for m-samples.
        theta: parameters (aka weights)
        alpha: learning rate
        num_iters: number of iterations
    """
    print("============Running Batch Perceptron===========")
    thresh = 0.001
    min = 1.0
    theta = np.zeros(X.shape[1])
    for i in range(num_iters):
        prediction = predict(X, theta)

        # errors = abs(y - prediction) # =1 for wrong prediction
        diffs = abs(y - prediction)  # =2 for wrong prediction
        error = [1 if err == 2 else 0 for err in diffs]
        grad_J = np.dot(error, X)
        theta = theta + alpha * grad_J

        error = sum(error)/X.shape[0]
        if error < min:
            # print("Θ",theta)
            epoch = i
            min = error
            best = theta
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (i, alpha, error))

        if all(abs(alpha * grad_J) < thresh):
            break
    print("Best Result:\n In epoch={}, found theta={} and error={}".format(epoch,best,min))
    print("============================================\n")

    return best


def batch_relaxation_with_margin(X, y, alpha, margin, num_iters):
    """
    Estimate perceptron weights using gradient descent with gradient of cost function J: =
    where y = falsely classified samples

    Parameters:
        -----------
        X: An [m x n] matrix containing the n-features for m-samples.
        y: A 1-d vector containing the true value for m-samples.
        theta: parameters (aka weights)
        alpha: learning rate
        margin: b in a.T*y > b
        num_iters: number of iterations
    """
    print("========Running Batch relaxation with margin========")
    thresh = 0.001
    min = 1
    theta = np.zeros(X.shape[1])
    for i in range(num_iters):
        prediction = predict(X, theta, margin)
        # print("pred",prediction)
        # error = abs(y - prediction) # =1 for wrong prediction

        diffs = abs(y - prediction)  # =2 for wrong prediction
        error = [1 if err == 2 else 0 for err in diffs]
        missclassified = np.dot(error, X).astype("float64")
        # print(missclassified)
        if all(missclassified) == 0:
            break
        grad_J = np.dot(missclassified.T, margin - np.dot(missclassified, theta) / np.dot(missclassified.T, missclassified))
        theta = theta - alpha * grad_J

        error = sum(error) / X.shape[0]
        if error < min:
            # print("Θ",theta)
            epoch = i
            min = error
            best = theta
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (i, alpha, error))

        if all(abs(alpha * grad_J) < thresh):
            break
    print("Best Result:\n In epoch={}, found theta={} and error={}".format(epoch, best, min))
    print("===================================================\n")
    return best


def least_squares(X, y):
    """
    Least squares using pseudoinverse
    (Minimum square error)
    """
    print("============Least square errors============")
    dot = np.dot(X.T, X).astype("float64")
    X_inv = np.linalg.inv(dot)
    theta = np.dot(np.dot(X_inv, X.T), y)
    prediction = predict(X, theta)
    # errors = abs(y - prediction) # =1 for wrong prediction
    # diffs = abs(y - prediction)  # =2 for wrong prediction
    # errors = [1 if err==2 else 0 for err in diffs]
    # error = sum(errors) / X.shape[0]
    error = np.mean(y != prediction)
    print("Calculated theta= {}. \nError= {}".format(theta,error))
    print("============================================\n")
    return theta


def least_mean_squares(X, y, alpha, num_iters):
    """
    Least Mean squares
    (Widrow-Hoff)
    """
    print("============Least mean squares============")
    thresh = 0.001
    min = 1
    theta = np.zeros(X.shape[1])
    for i in range(num_iters):
        grad_J = np.dot(X.T, (np.dot(X, theta) - y))
        theta = theta - alpha * grad_J

        prediction = predict(X, theta)
        # error = abs(y - prediction)
        # diffs = abs(y - prediction)  # =2 for wrong prediction
        # error = [1 if err == 2 else 0 for err in diffs]
        # error = sum(error) / X.shape[0]
        error = np.mean(y != prediction)
        if error < min:
            # print("Θ",theta)
            epoch = i
            min = error
            best = theta
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (i, alpha, error))

        if all(abs(alpha * grad_J) < thresh):
            break
    print("Best Result:\n In epoch={}, found theta={} and error={}".format(epoch, best, min))
    print("============================================\n")
    return best


def least_squares_HoKa(X, b, alpha, num_iters):
    """
    Least squares using Ho-kashyap method
    """
    print("============Least square errors (Ho-Kashyap)============")
    thresh_b = 1
    min = 1
    theta = np.zeros(X.shape[1])
    for i in range(num_iters):
        e = np.dot(X, theta) - b
        e_pos = (e + abs(e)) / 2
        b = b + 2 * alpha * e_pos
        dot = np.dot(X.T, X).astype("float64")
        X_inv = np.linalg.inv(dot)
        theta = np.dot(np.dot(X_inv, X.T), b)
        prediction = predict(X, theta)

        error = np.mean(b!=prediction)
        if error < min:
            # print("Θ",theta)
            epoch = i
            min = error
            best = theta
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (i, alpha, error))

        if all(abs(e) < thresh_b):
            break

    print("Best Result:\n In epoch={}, found theta={} and error={}".format(epoch, best, min))
    print("============================================\n")
    return best, b


# def leastSquaresMulti():

# def Kesler(X,y):


data = pd.read_csv(os.path.join(root_dir, "iris.csv"), header=None)
temp = data.to_numpy()

X_temp = temp[:, :-1]
Y_temp = temp[:, -1]

#Normalization
# X_norm = preprocessing.normalize(X_temp)
X_norm, mu, sigma = feature_standardize(X_temp)

# insert ones as a first dimension feature
X_norm = np.column_stack((np.ones((X_norm.shape[0])), X_norm))

# #Classes 0='Iris-setosa', 1='Iris-versicolor', 2='Iris-virginica'
# y_class = []
# for c in range(3):
#     for i in range(50):
#         y_class.append(c)
# y = np.array(y_class)
#
# #Classes 1='Iris-setosa', 0='Iris-versicolor' or 'Iris-virginica'
# y_setosa = []
# for i in range(50):
#     y_setosa.append(1)
# for i in range(100):
#     y_setosa.append(0)
# y_setosa = np.array(y_setosa)
#
# #Classes 0='Iris-setosa' or 'Iris-virginica', 1= 'Iris-versicolor'
# y_versicolor = np.array([i if i!=2 else 0 for i in y])
#
# #Classes 0='Iris-setosa' or 'Iris-versicolor', 1= 'Iris-virginica'
# y_virginica = []                                    
# for i in range(100):
#     y_virginica.append(0)
# for i in range(50):
#     y_virginica.append(1)
# y_virginica = np.array(y_virginica)

#Classes 0='Iris-setosa', 1='Iris-versicolor', 2='Iris-virginica'
y_class = []
for c in range(3):
    for i in range(50):
        y_class.append(c)
y = np.array(y_class)

#Classes 1='Iris-setosa', -1 ='Iris-versicolor' or 'Iris-virginica'
y_setosa = []
for i in range(50):
    y_setosa.append(1)
for i in range(100):
    y_setosa.append(-1)
y_setosa = np.array(y_setosa)

#Classes -1='Iris-setosa' or 'Iris-virginica', 1= 'Iris-versicolor'
y_versicolor = np.array([1 if i==1 else -1 for i in y])

#Classes -1='Iris-setosa' or 'Iris-versicolor', 1= 'Iris-virginica'
y_virginica = []                                    
for i in range(100):
    y_virginica.append(-1)
for i in range(50):
    y_virginica.append(1)
y_virginica = np.array(y_virginica)

"""
- Find a linear classifier seperating Iris Setosa from the others using:
  - batch perceptron
  - batch relaxation with margin
"""
print("A")
print("Finding linear classifier seperating setosa from the others:")

num_iters = 200

alpha = 0.01
theta = batch_perceptron(X_norm, y_setosa, alpha, num_iters)
plot_data(X_norm, y_setosa, title="A) Batch perceptron", xlabel="sepal length", ylabel="sepal width", theta=theta)

alpha = 0.01
margin = 0.5
theta = batch_relaxation_with_margin(X_norm, y_setosa, alpha, margin, num_iters)
plot_data(X_norm, y_setosa, title="A)Batch perceptron with margin", xlabel="sepal length", ylabel="sepal width", theta=theta)


"""
- Find a linear classifier seperating Iris Setosa from the others using:
  - least squares (with the use of the pseudotranspose)
  - least mean squares (Windrow-Hopf)
"""
print("B")
print("Finding linear classifier seperating setosa from the others:")

theta = least_squares(X_norm, y_setosa)
plot_data(X_norm, y_setosa, title="B) Least Squares", xlabel="sepal length", ylabel="sepal width", theta=theta)


alpha = 0.001
theta = least_mean_squares(X_norm, y_setosa, alpha, num_iters)
plot_data(X_norm, y_setosa, title="B) Least Mean Squares", xlabel="sepal length", ylabel="sepal width", theta=theta)


"""
- Find a linear classifier seperating versicolour and virginica using:
  - least squares (with the use of the pseudotranspose)
  - least squares (Ho-Kashyap method)
"""
print("C")
print("Finding linear classifier seperating versivolour and virginia:")

X_norm_2_3 = X_norm[50:, :]

#Classes -1='Iris-versicolor', 1= 'Iris-virginica'
y_2_3 = []
for i in range(50):
    y_2_3.append(-1)
for i in range(50):
    y_2_3.append(1)
y_2_3 = np.array(y_2_3)


theta = least_squares(X_norm_2_3, y_2_3)
plot_data(X_norm_2_3, y_2_3, title="C) Least squares", xlabel="sepal length", ylabel="sepal width", theta=theta)


alpha = 0.01
theta, b = least_squares_HoKa(X_norm_2_3, y_2_3, alpha, num_iters)
plot_data(X_norm_2_3, y_2_3, title="C) Least squares with Ho-Kashyap", xlabel="sepal length", ylabel="sepal width", theta=theta)


"""
- Find linear classifiers of all 3 classes using least squares(pseudotranspose) method and all features
"""
print("D")
print("Finding linear classifiers of all 3 classes:")

theta_multi = np.zeros((X_norm.shape[1],3))
theta_multi[:,0] = least_squares(X_norm, y_setosa)
theta_multi[:,1] = least_squares(X_norm, y_versicolor)
theta_multi[:,2] = least_squares(X_norm, y_virginica)

set = y_setosa == 1
vers = y_versicolor == 1
virg = y_virginica == 1

plt.title("D) All 3 linear classifiers")
plt.plot(X_norm[set, 1], X_norm[set, 2],'k*', lw=2, ms=10)
plt.plot(X_norm[vers, 1], X_norm[vers, 2],'ko', mfc='y', ms=8,mec='k',mew=1)
plt.plot(X_norm[virg, 1], X_norm[virg, 2],'r+', mfc='r', ms=8)
plt.plot(X_norm[:,1], np.matmul(X_norm[:, 0:2],theta_multi[0:2,0]), color='black', linewidth=2) # Line visualization
plt.plot(X_norm[:,1], np.matmul(X_norm[:, 0:2],theta_multi[0:2,1]), color='blue', linewidth=2) # Line visualization
plt.plot(X_norm[:,1], np.matmul(X_norm[:, 0:2],theta_multi[0:2,2]), color='red', linewidth=2) # Line visualization
plt.show()

"""
- Find linear classifiers of all 3 classes for spaces (1,2,3), (2,3,4) using least squares method
- Plot the hyperplanes 
"""
print("E")
print("Finding linear classifiers of all 3 classes for spaces (1,2,3), (2,3,4):")

X_norm_1_2_3 = X_norm[:, 0:4]
X_norm_2_3_4 = X_norm[:, 2:]
X_norm_2_3_4 = np.column_stack((np.ones((X_norm.shape[0])), X_norm_2_3_4))

print("(1,2,3)")
theta_multi = np.zeros((X_norm_1_2_3.shape[1], 3))
theta_multi[:, 0] = least_squares(X_norm_1_2_3, y_setosa)
theta_multi[:, 1] = least_squares(X_norm_1_2_3, y_versicolor)
theta_multi[:, 2] = least_squares(X_norm_1_2_3, y_virginica)

print("(2,3,4)")
theta_multi = np.zeros((X_norm_2_3_4.shape[1], 3))
theta_multi[:, 0] = least_squares(X_norm_2_3_4, y_setosa)
theta_multi[:, 1] = least_squares(X_norm_2_3_4, y_versicolor)
theta_multi[:, 2] = least_squares(X_norm_2_3_4, y_virginica)

plt.title("E) All 3 linear classifiers for space (2,3,4)")
plt.plot(X_norm[set, 3], X_norm[set, 4],'k*', lw=2, ms=10)
plt.plot(X_norm[vers, 3], X_norm[vers, 4],'ko', mfc='y', ms=8,mec='k',mew=1)
plt.plot(X_norm[virg, 3], X_norm[virg, 4],'r+', mfc='r', ms=8)
X2 = X_norm[:, 3:]
X2 = np.column_stack((np.ones((X_norm.shape[0])), X2))
plt.plot(X2[:, 1], np.matmul(X2[:, 0:2], [theta_multi[0,0], theta_multi[1,0]]), color='black', linewidth=3,
    label='Linear Regression')  # Line visualization
plt.plot(X2[:, 1], np.matmul(X2[:, 0:2], [theta_multi[0,1], theta_multi[2,1]]), color='blue', linewidth=3,
    label='Linear Regression')  # Line visualization
plt.plot(X2[:, 1], np.matmul(X2[:, 0:2], [theta_multi[0,2], theta_multi[3,2]]), color='red', linewidth=3,
    label='Linear Regression')  # Line visualization

plt.show()


#todo 
"""
- Find linear classifiers of all 3 classes using Kesler structure
"""
# print("F")
# print("Finding linear classifiers of all 3 classes:")

