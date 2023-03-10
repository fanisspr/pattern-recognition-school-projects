'''
Clustering methods for classifying samples from Iris dataset
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from fcmeans import FCM #fuzzy-c-means package
import isodata


def accuracy(cluster_labels: list[int], class_labels: list[str]) -> float:
    """
    Calculate the accuracy of a clustering algorithm on the Iris dataset.
    
    Parameters:
    cluster_labels: list[int] 
        The cluster labels predicted by the algorithm.
    class_labels: list[str]
        The true class labels for the samples.
    
    Returns:
    float: The accuracy of the algorithm, ranging from 0 to 100.
    """
    correct_pred = 0
    cluster_labels = list(cluster_labels)
    # each class gets a number(0,1,2) based on how many points has this number as a label
    seto = max(set(cluster_labels[0:50]), key=cluster_labels[0:50].count)
    vers = max(set(cluster_labels[50:100]), key=cluster_labels[50:100].count)
    virg = max(set(cluster_labels[100:]), key=cluster_labels[100:].count)
    for i in range(len(X)):
        if cluster_labels[i] == seto and class_labels[i] == 'Iris-setosa':
            correct_pred = correct_pred + 1
        if cluster_labels[i] == vers and class_labels[i] == 'Iris-versicolor' and vers != seto:
            correct_pred = correct_pred + 1
        if cluster_labels[i] == virg and class_labels[i] == 'Iris-virginica' and virg != seto and virg != vers:
            correct_pred = correct_pred + 1

    accuracy = (correct_pred / len(X)) * 100
    return accuracy


X, y = load_iris(return_X_y=True)

class_labels = []
for i in range(len(X)):
    if i < 50:
        class_labels.append('Iris-setosa')
    elif i < 100:
        class_labels.append('Iris-versicolor')
    else:
        class_labels.append('Iris-virginica')


# standardize dataset
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)


"""
K-Means
"""

# This will perform ten runs of the k-means algorithm on the data with a maximum of 100 iterations per run:
kmeans = KMeans(init="random",
                n_clusters=3,
                n_init=10,
                max_iter=100,
                random_state=3)

kmeans.fit(X)
y_kmeans = kmeans.predict(X)

error = 100 - accuracy(y_kmeans, class_labels)
print("K-means:")
# print(" ??ccuracy: {0:.4f}".format(accuracy))
print(" Error: {0:.4f}%".format(error))

""" 
fuzzy C-means
"""

fcmeans = FCM(n_clusters=3,
              max_iter=100,
              m=2,
              random_state=3)
fcmeans.fit(X)
y_fcmeans = fcmeans.predict(X)

error = 100 - accuracy(y_fcmeans, class_labels)
print("fuzzy C-means:")
print(" m=2:")
# print(" ??ccuracy: {0:.4f}".format(accuracy))
print("   Error: {0:.4f}%".format(error))


fcmeans = FCM(n_clusters=3,
              max_iter=100,
              m=6,
              random_state=3)
fcmeans.fit(X)
y_fcmeans = fcmeans.predict(X)

error = 100 - accuracy(y_fcmeans, class_labels)
print(" m=6:")
# print(" ??ccuracy: {0:.4f}".format(accuracy))
print("   Error: {0:.4f}%".format(error))


"""
ISODATA
clusters are split if standard deviation is big(data not similar) and merged if std is small (similar data)
"""