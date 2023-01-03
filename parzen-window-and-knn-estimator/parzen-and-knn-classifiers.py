import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neighbors import KernelDensity


def knn_classify(k: int, test_samples: np.ndarray, train_samples: np.ndarray, train_labels: list[int]) -> list[int]:
    """
    Classify test samples using the k-nearest neighbors (KNN) algorithm.

    Parameters
    ----------
    k : int
        The number of nearest neighbors to consider when making a prediction.
    test_samples : ndarray
        The test samples.
    train_samples : ndarray
        The training samples.
    train_labels : list[int]
        The labels for the training samples.

    Returns
    -------
    class_predictions : list[int]
        A list of class predictions, one for each test sample.
    """
    class_predictions = []
    for test_samp in test_samples:
         distances = np.abs(train_samples - test_samp)  #for higher Dims: np.sum(() ** 2, axis=1)
         retrieved_ids = np.argsort(distances)
         retrieved_ids = retrieved_ids[:k]

         neighbors_class = [train_labels[id] for id in retrieved_ids]
         prediction = max(set(neighbors_class), key= neighbors_class.count)
         class_predictions.append(prediction)

    return class_predictions


class ParzenWindow: 
    """
    A class for estimating the probability density of a 1D dataset 
    using the Parzen window method.

    Parameters
    ----------
    data : ndarray
        1D array of data points.
    """
    
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


    #Parzen window density estimation, with gaussian kernel
    def fit(self, bandwidth: float, kernel: str ='gaussian') -> KernelDensity:
        """
        Fit a kernel density model using the Parzen window method.

        Parameters
        ----------
        bandwidth : float
            Bandwidth of the kernel.
        kernel : str, optional
            Type of kernel to use. Default is 'gaussian'.

        Returns
        -------
        pwde : KernelDensity
            Fitted kernel density model.
        """
        pwde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(self.data.reshape(-1,1))
        return pwde


    #Evaluate the log density model on the points.
    def estimate(self, data: int, bandwidth: float) -> np.ndarray:
        """
        Estimate the probability density of the data using the kernel density model.

        Parameters
        ----------
        data : int
            data to use for estimating the density.
        bandwidth : float
            Bandwidth of the kernel.

        Returns
        -------
        pdf_values : ndarray
            Array of estimated probability density values.
        """
        pwde = self.fit(bandwidth=bandwidth)
        pdf_values = np.exp(pwde.score_samples(data.reshape(-1,1)))
        return pdf_values
        

    def plot(self, points_num: int, 
             bandwidth: float, 
             pdf_values: np.ndarray, 
             ax) -> None:  
        """
        Plot the estimated density of the data using the kernel density model.

        Parameters
        ----------
        points_num : int
            Number of points to use for estimating the density.
        bandwidth : float
            Bandwidth of the kernel.
        pdf_values : ndarray
            Array of probability density values to plot.
        ax : matplotlib Axes
            Axes object to use for plotting.

        Returns
        -------
        None
            The estimated density is plotted using matplotlib.
        """          
        points = np.linspace(-10, 10, points_num)
        ax.plot(points, pdf_values, 'r.')
        ax.set(title=f"N= {points_num}, h= {bandwidth}", 
                xlabel='x',
                ylabel='pdf')

    #Classify with parzen windows
    @staticmethod
    def classify(priors: list[float], *likelihoods: np.ndarray) -> list[int]:
        """
        Classify samples using the Bayes rule.

        Parameters
        ----------
        priors : list[float]
            A list of prior probabilities for each class.
        likelihoods : list[np.ndarray]
            A list of likelihoods for each class, 
            where each likelihood is a list of likelihood values for each sample.

        Returns
        -------
        predictions : list[int]
            A list of predicted class labels for each sample.
        """
        # likelihood = ParsenWindow(train_samples,test_samples,h)
        predictions = []
        for i in range(len(likelihoods[0])):
            post_p1 = likelihoods[0][i] * priors[0]
            post_p2 = likelihoods[1][i] * priors[1]
            post_p3 = likelihoods[2][i] * priors[2]
            prediction = max(post_p1,post_p2,post_p3)
            if (prediction == post_p1):
                predictions.append(1)
            elif(prediction == post_p2):
                predictions.append(2)
            else: predictions.append(3)
        return predictions


def pnn_train(X: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    """
    Train a Probabilistic Neural Network (PNN) classifier.

    Parameters
    ----------
    X : np.ndarray
        A numpy array of shape (n_samples, n_features) representing the input data.
    y : np.ndarray
        A numpy array of shape (n_samples,) representing the class labels.

    Returns
    -------
    weights : list[np.ndarray]
        A list of numpy arrays, where each array represents the weights for one sample in X.
    activations : list[int]
        A list of class labels, where each element represents the activation for one sample in X.
    """
    weights = []
    activations = []
    # activations = np.zeros(3, X.size)

    for i, train_samp in enumerate(X):
        #normalize
        train_samp = train_samp / abs(train_samp)
        #Weight training
        weights.append(train_samp)
        #Activation (Connection to known class y[i])
        activations.append(y[i])
    return weights, activations


def pnn_classify(X: np.ndarray, weights: list[np.ndarray], activations: list[int], h: float) -> list[int]:
    """
    Classify samples using a Probabilistic Neural Network (PNN).

    Parameters
    ----------
    X : np.ndarray
        A numpy array of shape (n_samples, n_features) representing the input data.
    weights : list[np.ndarray]
        A list of numpy arrays, where each array represents the weights for one sample in X.
    activations : list[int]
        A list of class labels, where each element represents the activation for one sample in X.
    h : float
        The spread constant for the Gaussian function used in the PNN.

    Returns
    -------
    predictions : list[int]
        A list of predicted class labels for each sample in X.
    """
    predictions = []

    for i, test_samp in enumerate(X):
        sum1 = 0
        sum2 = 0
        sum3 = 0
        net = [weight * test_samp for weight in weights]

        for j, cl in enumerate(activations):
            activation = np.exp((net[j] - 1) / (h**2))
            if (cl == 1):
                sum1 += activation
            elif (cl == 2):
                sum2 += activation
            elif (cl == 3):
                sum3 += activation

        prediction = max(sum1,sum2,sum3)
        if (prediction == sum1):
            predictions.append(1)
        elif (prediction == sum2):
            predictions.append(2)
        else:
            predictions.append(3)

    return predictions


def classification_accuracy(predictions: list[int], test_labels: list[int]) -> float:
    """
    Calculate the classification accuracy given predicted class labels and true class labels.

    Parameters
    ----------
    predictions : list[int]
        A list of predicted class labels.
    test_labels : list[int]
        A list of true class labels.

    Returns
    -------
    accuracy : float
        The classification accuracy, defined as the number of correct predictions divided by the total number of samples.
    """
    correct = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predictions[i]:
            correct += 1
    accuracy = correct / float(len(test_labels))
    # print("Classification accuracy: ", accuracy)
    return accuracy


def create_labels(N1: int, N2: int, N3: int) -> list[int]:
    """
    Create a list of class labels.

    Parameters
    ----------
    N1 : int
        The number of samples in class 1.
    N2 : int
        The number of samples in class 2.
    N3 : int
        The number of samples in class 3.

    Returns
    -------
    labels : list[int]
        A list of class labels, where each label is either 1, 2, or 3.
    """
    labels = []
    for i in range(1, N1+1):
        labels.append(1)
    for i in range(1, N2+1):
        labels.append(2)
    for i in range(1, N3+1):
        labels.append(3)
    return labels


'''
There are 3 classes having: P(x|ω1) = N(2, 0.5), P(x|ω2) = N(1,1), P(x|ω3) = N(3, 1.2)
and priors: P(ω1) = 0.5, P(ω2) = 0.3, P(ω3) = 0.2.
'''

'''
- Create random train sample with 100 samples(features) based on above distributions and priors.
- Create random test sample with 1000 features based on above distributions and priors
'''

# priors
p = [0.5, 0.3, 0.2]

# Create the training set according to the priors
Nx = 100
Nx1 = int(Nx * p[0])
Nx2 = int(Nx * p[1])
Nx3 = int(Nx * p[2])

# data with distributions: N(2, 0.5), N(1,1) and N(3, 1.2) and according to priors
dx1 = np.random.normal(2, 0.5, Nx1)
dx2 = np.random.normal(1, 1, Nx2)
dx3 = np.random.normal(3, 1.2, Nx3)

train_samples = np.concatenate((dx1,dx2,dx3), axis=None)

# create labels of classes = 1 or 2 or 3
train_labels = create_labels(Nx1, Nx2, Nx3)

# Create the testing set according to the priors
Ny = 1000
Ny1 = int(Ny * p[0])
Ny2 = int(Ny * p[1])
Ny3 = int(Ny * p[2])

# data with distributions: N(2, 0.5), N(1,1) and N(3, 1.2) and according to priors
dy1 = np.random.normal(2, 0.5, Ny1)
dy2 = np.random.normal(1, 1, Ny2)
dy3 = np.random.normal(3, 1.2, Ny3)

test_samples = np.concatenate((dy1,dy2,dy3), axis=None)

# create labels of classes = 1 or 2 or 3
test_labels = create_labels(Ny1, Ny2, Ny3)


'''
- Classify test samples using knn algorithm, for k=1,2,3, and calculate error.
- Compare error to that of the Bayesian classifier.
'''


# Classify with Knn:
for k in [1, 2, 3]:
    Knn_classes = knn_classify(k, test_samples, train_samples, train_labels)
    Acc = classification_accuracy(Knn_classes, test_labels)
    print(str(k)+"-nn accuracy: ", Acc)

# Finding the best k:
results = []
for k in range(1,50):
    Knn_classes = knn_classify(k, test_samples, train_samples, train_labels)
    acc = classification_accuracy(Knn_classes, test_labels)
    results.append(acc)
best_k = max(results)
best_k_ind = results.index(best_k)
print(str(best_k_ind) + "-nn accuracy: ", best_k)


# Bayesian classifier:
# Create a Gaussian Classifier
gnb = GaussianNB()

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
# Train the model using the training sets
gnb.fit(train_samples.reshape(-1,1), train_labels.reshape(-1,1))
# Predict the response for test dataset
y_pred = gnb.predict(test_samples.reshape(-1,1))
# Model Accuracy
print("Bayesian accuracy:", metrics.accuracy_score(test_labels, y_pred))


'''
- Use samples for classification using Parzen Windows. 
- Use atleast 4 different values for the bandwidth parameter h = σ (spread) 
  and choose the one with the best results
'''

# Classify with Parzen window
pw1 = ParzenWindow(dx1)
pw2 = ParzenWindow(dx2)
pw3 = ParzenWindow(dx3)
for h in [0.02, 0.1, 0.5, 1, 4]:
    likelihood1 = pw1.estimate(test_samples, h)
    likelihood2 = pw2.estimate(test_samples, h)
    likelihood3 = pw3.estimate(test_samples, h)
    parzen_classes = ParzenWindow.classify(p, likelihood1, likelihood2, likelihood3)
    acc = classification_accuracy(parzen_classes, test_labels)
    print("Parzen window accuracy, for h= "+str(h)+": ", acc)


# Finding the best h:
results = []
for i in range(1,100):
    h = 0.01 * i
    likelihood1 = pw1.estimate(test_samples, h)
    likelihood2 = pw2.estimate(test_samples, h)
    likelihood3 = pw3.estimate(test_samples, h)
    parzen_classes = ParzenWindow.classify(p, likelihood1, likelihood2, likelihood3)
    acc = classification_accuracy(parzen_classes, test_labels)
    results.append(acc)
best_h = max(results)
best_h_ind = 0.01 * results.index(best_h)
print("Parzen window accuracy, for h= "+str(best_h_ind)+": ", best_h)


'''
- Use samples for classification with Parzen Windows/Probabilistic Neural Networks.
- Use 4 different values for the bandwidth parameter h = σ (spread) 
'''

weights, activations = pnn_train(train_samples, train_labels)
for h in [0.02, 0.1, 0.5, 1, 4]:
    Pnn_classes = pnn_classify(test_samples, weights, activations, h)
    Acc = classification_accuracy(Pnn_classes, test_labels)
    print("Parzen/Pnn accuracy, for h= "+str(h)+": ", Acc)