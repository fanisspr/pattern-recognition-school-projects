'''
The Iris data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). 
Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
Iris Setosa is the only one that is linearly seperable from the others

In this project, SVM linear and non-linear classifiers are used to classify samples from the iris dataset
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler


root_dir = os.path.relpath(os.path.dirname(__file__))
plots_dir = os.path.join(root_dir, 'plots')
if not os.path.exists(plots_dir):
    os.mkdir(plots_dir)


def plot_data(clf: svm.SVC, X: np.ndarray, y: np.ndarray, title: str, **kwargs):
    '''
    create a scatter plot of data points with two classes, 
    and plot the decision function of a classifier.

    Parameters:
    ----------
    clf: sklearn.svm.SVC
        a classifier object with a fit(X, y) and predict(X) method
    X: ndarray
        feature matrix with shape (n_samples, n_features)
    y: ndarray
        true labels with shape (n_samples,)
    title: str
        title of the plot
    kwargs: additional parameters
    '''
    pos = y == 1
    neg = y == -1
    plt.figure()
    plt.title(title)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.plot(X[pos, 0], X[pos, 1], 'bo', label=kwargs.get('first_class_label', None))
    plt.plot(X[neg, 0], X[neg, 1], 'ro', label=kwargs.get('second_class_label', None))
    plot_svc_decision_function(clf, **kwargs)
    plt.legend()
    if kwargs.get('save_to_file', None):
        plt.savefig(os.path.join(plots_dir, kwargs.get('save_to_file')))
    plt.show()

def plot_data_multi(clfs: list[svm.SVC], X: np.ndarray, y: np.ndarray, title: str, **kwargs):
    """
    create a scatter plot of data points of all 3 classes, 
    and plot the 3 decision functions that seperate each class from the others (one-vs-all stratregy)

    Parameters:
    ----------
    clf: list[sklearn.svm.SVC]
        a list of classifier objects with a fit(X, y) and predict(X) method
    X: ndarray
        feature matrix with shape (n_samples, n_features)
    y: ndarray
        true labels with shape (n_samples,)
    title: str
        title of the plot
    kwargs: additional parameters to pass
    """
    zero = y == 0
    one = y == 1
    two = y == 2
    plt.figure()
    plt.title(title)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.plot(X[zero, 0], X[zero, 1], 'bo', label=kwargs.get('first_class_label', None))
    plt.plot(X[one, 0], X[one, 1], 'ro', label=kwargs.get('second_class_label', None))
    plt.plot(X[two, 0], X[two, 1], 'ko', label=kwargs.get('third_class_label', None))
    plot_svc_decision_function(clfs[0], color='b', **kwargs)
    plot_svc_decision_function(clfs[1], color='r', **kwargs)
    plot_svc_decision_function(clfs[2], **kwargs)
    plt.legend()
    if kwargs.get('save_to_file', None):
        plt.savefig(os.path.join(plots_dir, kwargs.get('save_to_file')))
    plt.show()

def plot_svc_decision_function(model: svm.SVC,
                               ax=None,
                               plot_support: bool=True,
                               color: str='k',
                               levels: list[int]=[-1, 0, 1],
                               linestyles: list[str]=['--', '-', '--'],
                               **kwargs) -> None:
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


X, y = load_iris(return_X_y=True)

# Split data into train, test
X_train_all = np.concatenate([X[i:(i+40), :] for i in [0,50,100]])
X_test_all = np.concatenate([X[(i+40):(i+50), :] for i in [0,50,100]])
X_train_3 = np.hstack((X_train_all[:, 0].reshape(-1,1), X_train_all[:, 1].reshape(-1,1), X_train_all[:, 3].reshape(-1,1)))
X_test_3 = np.hstack((X_test_all[:, 0].reshape(-1,1), X_test_all[:, 1].reshape(-1,1), X_test_all[:, 3].reshape(-1,1)))
y_train = np.concatenate([y[i:(i+40)] for i in [0,50,100]])
y_test = np.concatenate([y[(i+40):(i+50)] for i in [0,50,100]])

# Classes 1='Iris-setosa', -1 ='Iris-versicolor' or 'Iris-virginica'
y_setosa_train = np.array([1 if i==0 else -1 for i in y_train])
y_setosa_test = np.array([1 if i==0 else -1 for i in y_test])

# Classes 1='Iris-versicolor', -1 ='Iris-setosa' or 'Iris-virginica'
y_versicolor_train = np.array([1 if i==1 else -1 for i in y_train])
y_versicolor_test = np.array([1 if i==1 else -1 for i in y_test])

# Classes 1='Iris-virginica', -1 ='Iris-versicolor' or 'Iris-setosa'
y_virginica_train = np.array([1 if i==2 else -1 for i in y_train])
y_virginica_test = np.array([1 if i==2 else -1 for i in y_test])


for X_train, X_test, features in zip([X_train_3, X_train_all], [X_test_3, X_test_all], ["(1, 2, 4)", "(1, 2, 3, 4)"]):
    print("======================================================")
    print("Using features {}:\n". format(features))

    '''
    2 classes problem (Versicolor vs others):
    '''

    """ 
    - Classify data using a linear SVM.
    """
    print("A")
    print("Classifying with linear SVM:\n")

    C = 100
    # Linear SVM
    lin_clf = svm.SVC(kernel='linear', C=C)
    lin_clf.fit(X_train, y_versicolor_train)


    # calculate accuracy
    accuracy = lin_clf.score(X_train, y_versicolor_train)
    print('\tTraining accuracy: {0:.2f}'.format(accuracy))
    scores = cross_val_score(lin_clf, X_train, y_versicolor_train, cv=5)
    print("\tValidation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    accuracy = lin_clf.score(X_test, y_versicolor_test)
    print('\tTesting accuracy: {0:.2f}'.format(accuracy))

    # linear SVM for 2D plotting
    clf = svm.SVC(kernel='linear', C=C)
    clf.fit(X_train[:,:2], y_versicolor_train)
    plot_data(clf, 
                X_train, 
                y_versicolor_train, 
                title='Linear kernel SVM',
                first_class_label='Iris Versicolor',
                second_class_label='Other classes',
                save_to_file='linear-svm.png')

    """ 
    - Classify data using non-linear SVMs.    
    """
    print("\nB")
    print("Classifying with various non-linear SVMs:\n")

    C = 100
    models = (svm.SVC(kernel='poly', degree=2, C=C),
              svm.SVC(kernel='poly', degree=3, C=C),
              svm.SVC(kernel='poly', degree=4, C=C),
              svm.SVC(kernel='rbf', C=C))
    models = (clf.fit(X_train, y_versicolor_train) for clf in models)

    # title for the plots
    titles = ('SVM with polynomial kernel of degree 2',
              'SVM with polynomial kernel of degree 3',
              'SVM with polynomial kernel of degree 4',
              'SVM with RBF kernel')

    # calculate accuracy
    for clf, title in zip(models, titles):
        print(title)
        accuracy = clf.score(X_train, y_versicolor_train)
        print('\tTraining accuracy: {0:.2f}'.format(accuracy))
        scores = cross_val_score(clf, X_train, y_versicolor_train, cv=5)
        print("\tValidation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        accuracy = clf.score(X_test, y_versicolor_test)
        print('\tTesting accuracy: {0:.2f}'.format(accuracy))


    # SVMs for 2D plotting
    models = (svm.SVC(kernel='poly', degree=2, C=C),
              svm.SVC(kernel='poly', degree=3, C=C),
              svm.SVC(kernel='poly', degree=4, C=C),
              svm.SVC(kernel='rbf', C=C))

    for clf, title in zip(models, titles):
        clf.fit(X_train[:, :2], y_versicolor_train)
        plot_data(clf, 
                    X_train, 
                    y_versicolor_train, 
                    title=title,
                    first_class_label='Iris Versicolor',
                    second_class_label='Other classes',
                    save_to_file=title)


    '''
    3 classes problem:
    '''

    """ 
    - Classify data using a linear SVM.    
    """
    print("\nC")
    print("Multi-class classification with linear SVM:\n")

    C = 100
    # Linear SVM
    lin_clf = svm.SVC(kernel='linear', C=C)
    lin_clf.fit(X_train, y_train)

    # calculate accuracy
    accuracy = lin_clf.score(X_train, y_train)
    print('\tTraining accuracy: {0:.2f}'.format(accuracy))
    scores = cross_val_score(lin_clf, X_train, y_train, cv=5)
    print("\tValidation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    accuracy = lin_clf.score(X_test, y_test)
    print('\tTesting accuracy: {0:.2f}'.format(accuracy))

    # linear SVM for 2D plotting
    clf_setosa = svm.SVC(kernel='linear', C=C).fit(X_train[:,:2], y_setosa_train)
    clf_versic = svm.SVC(kernel='linear', C=C).fit(X_train[:,:2], y_versicolor_train)
    clf_virgin = svm.SVC(kernel='linear', C=C).fit(X_train[:,:2], y_virginica_train)
    clfs = [clf_setosa, clf_versic, clf_virgin]
    plot_data_multi(clfs, 
                    X_train, 
                    y_train, 
                    title='3 Linear kernel SVMs',
                    first_class_label='Iris Setosa',
                    second_class_label='Iris Versicolor',
                    third_class_label='Iris Virginica',
                    save_to_file='3-linear-svms')

    """ 
    - Classify data using non-linear SVMs.   
    """
    print("\nD")
    print("Multi-class classification with various non-linear SVMs:\n")

    C = 100
    models = (svm.SVC(kernel='poly', degree=2, C=C),
              svm.SVC(kernel='poly', degree=3, C=C),
              svm.SVC(kernel='poly', degree=4, C=C),
              svm.SVC(kernel='rbf', C=C))
    models = (clf.fit(X_train, y_train) for clf in models)

    # title for the plots
    titles = ('3 SVMs with polynomial kernel of degree 2',
              '3 SVMs with polynomial kernel of degree 3',
              '3 SVMs with polynomial kernel of degree 4',
              '3 SVMs with RBF kernel')

    # calculate accuracy
    for clf, title in zip(models, titles):
        print(title)
        accuracy = clf.score(X_train, y_train)
        print('\tTraining accuracy: {0:.2f}'.format(accuracy))
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        print("\tValidation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        accuracy = clf.score(X_test, y_test)
        print('\tTesting accuracy: {0:.2f}'.format(accuracy))


    # SVMs for 2D plotting
    models = ("svm.SVC(kernel='poly', degree=2, C=C)",
              "svm.SVC(kernel='poly', degree=3, C=C)",
              "svm.SVC(kernel='poly', degree=4, C=C)",
              "svm.SVC(kernel='rbf', C=C)")

    for clf, title in zip(models, titles):
        clf_setosa = eval(clf).fit(X_train[:, :2], y_setosa_train)
        clf_versic = eval(clf).fit(X_train[:, :2], y_versicolor_train)
        clf_virgin = eval(clf).fit(X_train[:, :2], y_virginica_train)
        clfs = [clf_setosa, clf_versic, clf_virgin]
        plot_data_multi(clfs, 
                        X_train, 
                        y_train, 
                        title=title, 
                        first_class_label='Iris Setosa',
                        second_class_label='Iris Versicolor',
                        third_class_label='Iris Virginica',
                        save_to_file=title,
                        levels=[0], 
                        linestyles=['-'])
