'''
In this project, the XOR problem is solved using a non-llinear SVM with a polynomial kernel with degree of 2
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm


root_dir = os.path.relpath(os.path.dirname(__file__))
plots_dir = os.path.join(root_dir, 'plots')
if not os.path.exists(plots_dir):
    os.mkdir(plots_dir)


def plot_svc_decision_function(model: svm.SVC,
                               ax=None,
                               plot_support: bool=True,
                               color: str='k',
                               levels: list[int]=[-1, 0, 1],
                               linestyles: list[str]=['--', '-', '--']) -> None:
    """
    Plot the decision function for a 2D SVC.
    
    Parameters:
    ----------
    model (SVC): A trained 2D SVC model.
    ax (matplotlib.axes.Axes, optional): The axes object to plot on.
        If not specified, the current axes object will be used.
    plot_support (bool, optional): Whether to plot the support vectors.
        Defaults to True.
    color (str, optional): The color of the decision boundary and margins.
        Defaults to 'k' (black).
    levels (List[int], optional): The levels to plot the decision function.
        Defaults to [-1, 0, 1].
    linestyles (List[str], optional): The line styles to use for the
        decision boundary and margins. Defaults to ['--', '-', '--'].
    """
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors=color,
               levels=levels, alpha=0.5,
               linestyles=linestyles)

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
plt.plot(X[pos, 0], X[pos, 1], 'bo', label='1')
plt.plot(X[neg, 0], X[neg, 1], 'ro', label='0')
plot_svc_decision_function(clf)
plt.title('XOR problem classification with SVM')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'xor-problem-classification.png'))
plt.show()

print(f'SVM\'s non-linear equation coefficients: {clf.decision_function(X)}')
print(f'Predict (0,0): {clf.predict([[0, 0]])}')
print(f'Predict (1,0): {clf.predict([[1, 0]])}')
print(f'Predict (0,1): {clf.predict([[0, 1]])}')
print(f'Predict (1,1): {clf.predict([[1, 1]])}')