import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from fcmeans import FCM #fuzzy-c-means package
import isodata


def accuracy(cluster_labels, class_labels):
    correct_pred = 0
    # print(cluster_labels)
    cluster_labels = list(cluster_labels)
    #each class gets a number(0,1,2) based on how many points has this number as a label
    seto = max(set(cluster_labels[0:50]), key=cluster_labels[0:50].count)
    vers = max(set(cluster_labels[50:100]), key=cluster_labels[50:100].count)
    virg = max(set(cluster_labels[100:]), key=cluster_labels[100:].count)
    # print(seto)
    # print(vers)
    # print(virg)
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
κάθε iteration οδηγει σε διαφορετικη ομαδοποιηση, λογω της τυχαιας αρχικοποιησης των κεντρων.
οποτε η αρχικοποιηση παιζει σημαντικο ρολο
"""

# This will perform ten runs of the k-means algorithm on the data with a maximum of 300 iterations per run:
kmeans = KMeans(init="random",
                n_clusters=3,
                n_init=10,
                max_iter=100,
                random_state=3)

kmeans.fit(X)
y_kmeans = kmeans.predict(X)

error = 100 - accuracy(y_kmeans, class_labels)
print("K-means:")
# print(" Αccuracy: {0:.4f}".format(accuracy))
print(" Error: {0:.4f}%".format(error))

""" 
fuzzy C-means. 
Introduces Membership fucntion: Probability of a point to belong to a cluster
fuzzifier (m). If m = 1 => knn
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
# print(" Αccuracy: {0:.4f}".format(accuracy))
print("   Error: {0:.4f}%".format(error))


fcmeans = FCM(n_clusters=3,
              max_iter=100,
              m=6,
              random_state=3)
fcmeans.fit(X)
y_fcmeans = fcmeans.predict(X)

error = 100 - accuracy(y_fcmeans, class_labels)
print(" m=6:")
# print(" Αccuracy: {0:.4f}".format(accuracy))
print("   Error: {0:.4f}%".format(error))


"""
ISODATA
clusters με ανομοια δεδομενα διαρουνται και clusters με ομοια ενωνονται.
clusters are split if standard deviation is big(data not similar) and merged if std is small (similar data)
"""