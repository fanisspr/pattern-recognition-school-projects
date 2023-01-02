import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neighbors import KernelDensity as kd


#Knn classifier algorithm
def Knn(k, test_samples, train_samples, train_labels):
    class_predictions = []
    for test_samp in test_samples:
         distances = np.abs(train_samples - test_samp)  #for higher Dims: np.sum(() ** 2, axis=1) #Euclidean Distance = sqrt(sum i to N (x1_i – x2_i)^2)
         retrieved_ids = np.argsort(distances)
         retrieved_ids = retrieved_ids[:k]

         neighbors_class = [train_labels[id] for id in retrieved_ids]
         prediction = max(set(neighbors_class), key= neighbors_class.count)
         class_predictions.append(prediction) # Make a classification prediction with neighbors

    return class_predictions

#Create pdf with parsen window
def ParsenWindow(X, y, h):
    #Parsen window density estimation
    pwde = kd(kernel='gaussian', bandwidth = h).fit(X.reshape(-1,1))

    #Evaluate the log density model on the data.
    pdf_vals = np.exp(pwde.score_samples(y.reshape(-1,1)))

    return pdf_vals

#Classify with parzen windows
def Parsen_classify(test_samples, dx1, dx2, dx3, p, h):
    # likelihood = ParsenWindow(train_samples,test_samples,h)
    likelihood1 = ParsenWindow(dx1, test_samples, h)
    likelihood2 = ParsenWindow(dx2, test_samples, h)
    likelihood3 = ParsenWindow(dx3, test_samples, h)
    predictions = []
    for i in range(0, Ny):
        post_p1 = likelihood1[i] * p[0]
        post_p2 = likelihood2[i] * p[1]
        post_p3 = likelihood3[i] * p[2]
        prediction = max(post_p1,post_p2,post_p3)
        if (prediction == post_p1):
            predictions.append(1)
        elif(prediction == post_p2):
            predictions.append(2)
        else: predictions.append(3)

    return predictions

#Train the Pnn model
def Pnn_train(X,y):
    w = []
    a = []
    # a = np.zeros(3, X.size)

    for (i,train_samp) in enumerate(X):
        #normalize
        train_samp = train_samp / abs(train_samp)
        #Weight training
        w.append(train_samp)
        #Activation (Connection to known class y[i])
        a.append(y[i])

        #alternative:
        # if(train_labels[i] == 1):
        #     a[0][i] = 1
        # elif(train_labels[i] == 2):
        #     a[1][i] = 1
        # else:
        #     a[2][i] = 1

    return (w,a)

#Classify with parzen Pnn
def Pnn_classify(X, w, a, h):
    predictions = []

    for (i,test_samp) in enumerate(X):
        sum1 = 0
        sum2 = 0
        sum3 = 0
        net = [weight * test_samp for weight in w]

        for (j,cl) in enumerate(a):
            activation = np.exp((net[j] - 1) / (h**2))
            if (cl == 1):
                sum1 += activation
            if (cl == 2):
                sum2 += activation
            if (cl == 3):
                sum3 += activation

        prediction = max(sum1,sum2,sum3)
        if (prediction == sum1):
            predictions.append(1)
        elif (prediction == sum2):
            predictions.append(2)
        else:
            predictions.append(3)

    return predictions

#Classification accuracy
def classification_accuracy(predictions, test_labels):
    correct = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predictions[i]:
            correct += 1
    accuracy = correct / float(len(test_labels))
    # print("Classification accuracy: ", accuracy)
    return accuracy



#Α

p = [0.5, 0.3, 0.2]

#Create the training sets
Nx = 100
Nx1 = int(Nx * p[0])
Nx2 = int(Nx * p[1])
Nx3 = int(Nx * p[2])

dx1 = np.random.normal(2, 0.5, Nx1)
dx2 = np.random.normal(1, 1, Nx2)
dx3 = np.random.normal(3, 1.2, Nx3)

train_samples = np.concatenate((dx1,dx2,dx3), axis=None)

train_labels = []
for i in range(1, Nx1+1):
    train_labels.append(1)
for i in range(1, Nx2+1):
    train_labels.append(2)
for i in range(1, Nx3+1):
    train_labels.append(3)

#Create the testing sets
Ny = 1000
Ny1 = int(Ny * p[0])
Ny2 = int(Ny * p[1])
Ny3 = int(Ny * p[2])

dy1 = np.random.normal(2, 0.5, Ny1)
dy2 = np.random.normal(1, 1, Ny2)
dy3 = np.random.normal(3, 1.2, Ny3)

test_samples = np.concatenate((dy1,dy2,dy3), axis=None)

test_labels = []
for i in range(1, Ny1+1):
    test_labels.append(1)
for i in range(1, Ny2+1):
    test_labels.append(2)
for i in range(1, Ny3+1):
    test_labels.append(3)



#Β


#Classify with Knn:
for k in [1, 2, 3]:
    Knn_classes = Knn(k, test_samples, train_samples, train_labels)
    Acc = classification_accuracy(Knn_classes, test_labels)
    print(str(k)+"-nn accuracy: ", Acc)

#Finding the best k:
results = []
for k in range(1,50):
    Knn_classes = Knn(k, test_samples, train_samples, train_labels)
    acc = classification_accuracy(Knn_classes, test_labels)
    results.append(acc)
best_k = max(results)
best_k_ind = results.index(best_k)
print(str(best_k_ind) + "-nn accuracy: ", best_k)


#Bayesian classifier:
#Create a Gaussian Classifier
gnb = GaussianNB()

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
#Train the model using the training sets
gnb.fit(train_samples.reshape(-1,1), train_labels.reshape(-1,1))

#Predict the response for test dataset
y_pred = gnb.predict(test_samples.reshape(-1,1))

# Model Accuracy
print("Bayesian accuracy:",metrics.accuracy_score(test_labels, y_pred))



#Γ

#Classify with Parsen window
for h in [0.02, 0.1, 0.5, 1, 4]:
    parsen_classes = Parsen_classify(test_samples, dx1, dx2, dx3, p, h)
    acc = classification_accuracy(parsen_classes, test_labels)
    print("Parsen accuracy, for h= "+str(h)+": ", acc)


#Finding the best h:
results = []
for i in range(1,100):
    h = 0.01 * i
    parsen_classes = Parsen_classify(test_samples, dx1, dx2, dx3, p, h)
    acc = classification_accuracy(parsen_classes, test_labels)
    results.append(acc)
best_h = max(results)
best_h_ind = 0.01 * results.index(best_h)
print("Parsen accuracy, for h= "+str(best_h_ind)+": ", best_h)



#Δ


(w,a) = Pnn_train(train_samples,train_labels)
for h in [0.02, 0.1, 0.5, 1, 4]:
    Pnn_classes = Pnn_classify(test_samples, w, a, h)
    Acc = classification_accuracy(Pnn_classes, test_labels)
    print("Parsen/Pnn accuracy, for h= "+str(h)+": ", Acc)


