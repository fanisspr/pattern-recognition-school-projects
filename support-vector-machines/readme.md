# Xor problem
In this project, the XOR problem is solved using a non-llinear SVM with a polynomial kernel with degree of 2.

## Results:
We manage to classify 2 categories that are non-linear in this 2d space, 
using the kernel trick and not needing to use new features like x^2, that would cost more computationally

# Iris dataset SVM classifiers
The Iris data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). 
Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
Iris Setosa is the only one that is linearly seperable from the others

In this project, SVM linear and non-linear classifiers are used to classify samples from the iris dataset.

The following tasks are being executed:
- Using features (1,2,4):
    - For a 2 classes problem (Versicolor vs others):
        - Classify data using a linear SVM.
        - Classify data using non-linear SVMs.
        - Plot data using features sepal length and width
    - For a 3 classes problem (one vs all):
        - Classify data using a linear SVM.
        - Classify data using non-linear SVMs.
        - Plot data using features sepal length and width
- Repeat with features (1,2,3,4)

## Results:
Linear SVMs are not so good at classifying when the data is not that linearly separable. 
We can adjust their C parameter (the bigger, the 'harder' for datapoints to enter the margin) to increase their accuracy, 
but they cannot compare to non-linear SVMs. 
Regarding non-linear SVMs, using higher degree of polynomial kernel will not necessarily yield better results. 

One-vs-all strategy:
One way to improve accuracy is to use the one-vs-all strategy. By using 3 SVMs to seperate our 3 classes, 
we achieve perfect accuracy.

By using all 4 of the features, the training and validation accuracy is further increased.


Using features (1, 2, 4):

A
Classifying with linear SVM:

        Training accuracy: 0.74
        Validation accuracy: 0.68 (+/- 0.20)
        Testing accuracy: 0.73

B
Classifying with various non-linear SVMs:

SVM with polynomial kernel of degree 2
        Training accuracy: 0.96
        Validation accuracy: 0.96 (+/- 0.13)
        Testing accuracy: 0.97
SVM with polynomial kernel of degree 3
        Training accuracy: 0.93
        Validation accuracy: 0.93 (+/- 0.13)
        Testing accuracy: 0.97
SVM with polynomial kernel of degree 4
        Training accuracy: 0.95
        Validation accuracy: 0.92 (+/- 0.16)
        Testing accuracy: 0.97
SVM with RBF kernel
        Training accuracy: 0.96
        Validation accuracy: 0.92 (+/- 0.11)
        Testing accuracy: 1.00

C
Multi-class classification with linear SVM:

        Training accuracy: 0.93
        Validation accuracy: 0.94 (+/- 0.10)
        Testing accuracy: 1.00

D
Multi-class classification with various non-linear SVMs:

3 SVMs with polynomial kernel of degree 2
        Training accuracy: 0.94
        Validation accuracy: 0.93 (+/- 0.08)
        Testing accuracy: 1.00
3 SVMs with polynomial kernel of degree 3
        Training accuracy: 0.94
        Validation accuracy: 0.93 (+/- 0.08)
        Testing accuracy: 1.00
3 SVMs with polynomial kernel of degree 4
        Training accuracy: 0.95
        Validation accuracy: 0.93 (+/- 0.08)
        Testing accuracy: 1.00
3 SVMs with RBF kernel
        Training accuracy: 0.96
        Validation accuracy: 0.95 (+/- 0.08)
        Testing accuracy: 1.00
======================================================
Using features (1, 2, 3, 4):

A
Classifying with linear SVM:

        Training accuracy: 0.75
        Validation accuracy: 0.69 (+/- 0.18)
        Testing accuracy: 0.77

B
Classifying with various non-linear SVMs:

SVM with polynomial kernel of degree 2
        Training accuracy: 0.97
        Validation accuracy: 0.93 (+/- 0.17)
        Testing accuracy: 0.97
SVM with polynomial kernel of degree 3
        Training accuracy: 0.97
        Validation accuracy: 0.92 (+/- 0.19)
        Testing accuracy: 0.97
SVM with polynomial kernel of degree 4
        Training accuracy: 0.99
        Validation accuracy: 0.91 (+/- 0.21)
        Testing accuracy: 0.97
SVM with RBF kernel
        Training accuracy: 0.97
        Validation accuracy: 0.93 (+/- 0.16)
        Testing accuracy: 1.00

C
Multi-class classification with linear SVM:

        Training accuracy: 0.97
        Validation accuracy: 0.97 (+/- 0.07)
        Testing accuracy: 1.00

D
Multi-class classification with various non-linear SVMs:

3 SVMs with polynomial kernel of degree 2
        Training accuracy: 0.97
        Validation accuracy: 0.97 (+/- 0.07)
        Testing accuracy: 1.00
3 SVMs with polynomial kernel of degree 3
        Training accuracy: 0.97
        Validation accuracy: 0.97 (+/- 0.06)
        Testing accuracy: 1.00
3 SVMs with polynomial kernel of degree 4
        Training accuracy: 0.98
        Validation accuracy: 0.94 (+/- 0.07)
        Testing accuracy: 1.00
3 SVMs with RBF kernel
        Training accuracy: 0.98
        Validation accuracy: 0.96 (+/- 0.09)
        Testing accuracy: 1.00