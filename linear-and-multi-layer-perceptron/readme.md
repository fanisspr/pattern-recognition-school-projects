# Linear Perceptron

A linear perceptron is used to classify samples from the iris dataset.

- Data is split between train (120 samples) and test (30 samples) sets.
- Data is standardized and shuffled.
- 30% of the train set is used for validation.

## Results

Linear Perceptron accuracy:
    Train accuracy: 0.8833
    Test accuracy: 0.9667

# Multi-layer Perceptron

A multi-layer perceptron is used to classify samples from the iris dataset.

- Data is split between train (90 samples), validation (30 samples) and test (30 samples) sets.
- Data is shuffled and then given in batches.
- Sigmoid activation function for hidden layers and softmax for output neurons.

## Results

The more neurons the model has, the faster and easier it gets trained on the train data.
The linear perceptron cannot get trained that easily on the non-linearly separable data, therefore
it only reaches a train accuracy of 88%.

As is evident by the plots, the train accuracy increases as the number of neurons increases.
That does not necessarily mean that the model gets better at classifying new samples.

We can observe that the test accuracy remains the same for all the cases, because we have few and
easily separable features with the test loss varying.

In general, for a particular dataset, an ideal number of neurons must be found in order neither to
underfit nor overfit the data.

Test metrics for model with hidden layer sizes (2,):

        Test loss: 0.0916, Test Acc.: 0.9667

Test metrics for model with hidden layer sizes (5,):

        Test loss: 0.0803 Test Acc.: 0.9667

Test metrics for model with hidden layer sizes (10,):

        Test loss: 0.0655 Test Acc.: 0.9667

Test metrics for model with hidden layer sizes (5, 5):

        Test loss: 0.0677 Test Acc.: 0.9667    

Test metrics for model with hidden layer sizes (10, 5):

        Test loss: 0.0616 Test Acc.: 0.9667    

Test metrics for model with hidden layer sizes (10, 10):

        Test loss: 0.0971 Test Acc.: 0.9667    

Test metrics for model with hidden layer sizes (20, 20):

        Test loss: 0.0730 Test Acc.: 0.9667  

# Iris dataset

The Iris data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor).
Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
Iris Setosa is the only one that is linearly seperable from the others
