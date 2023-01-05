# Unsupervised Clustering

Clustering methods for classifying samples from Iris dataset.

- Kmeans clustering: Every iteration leads to different clusters, due to the random initialization of the centers. Initialization plays a big role on the final cluustering.
- Fuzzy C-means clustering: Introduces membership function: Probability of a point belonging in a cluster. Parameter m controls how fuzzy a cluster is. The bigger, the fuzzier. If fuzzifier m: = 1, then it is the same clustering as Kmeans

# Results

K-means algorithm chooses the best of the 10 runs (n_init=10) of 100 iterations (max_iter=100).
The parameter m of the fuzzy c-means algorithm is chosen with common values (m=2, m=6).
The fuzzy C-means has a lower error as is expected:

K-means:

    Error: 16.6667%

fuzzy C-means:

    m=2:
        Error: 16.0000%
    m=6:
        Error: 14.0000%

# Iris dataset

The Iris data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor).
Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
Iris Setosa is the only one that is linearly seperable from the others
