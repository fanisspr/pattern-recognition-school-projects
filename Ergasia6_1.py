import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    # x = np.linspace(-5, 5, 30)
    # y = np.linspace(-5, 5, 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


data = np.array([[0, 0, 0], [0,1, 1], [1,0, 1], [1,1, 0]])
X = data[:, :-1]
Y = data[:, -1]

clf = svm.SVC(kernel='poly',degree=2, coef0=0)
clf.fit(X, Y)

pos = Y == 1
neg = Y == 0
plt.figure()
plt.plot(X[pos, 0], X[pos, 1], 'bo')
plt.plot(X[neg, 0], X[neg, 1], 'ro')
plot_svc_decision_function(clf)
plt.show()

print(clf.decision_function(X))
print(clf.predict([[0, 0]]))
print(clf.predict([[1, 0]]))
print(clf.predict([[0, 1]]))
print(clf.predict([[1, 1]]))
