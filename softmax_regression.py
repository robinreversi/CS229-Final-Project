#
# This program runs softmax regression on a dataset
#
# Authors: Vince, Alex, Robin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

import mnist_data


def getLoss(W, X, Y, lam):
    """
    W = weight matrix where each entry column represents the kth category's
        weight for that feature
    X = data, m x n matrix
    Y = m x 1 vector, with each entry between 0 and k, to later be converted
        into a one-hot matrix
    lam = regularization parameter

    returns the loss and gradient of training (currently batch GD)
    """
    # num training examples
    m = X.shape[0]
    # convert int to one-hot representation
    # m x k matrix
    one_hot_Y = convToOneHot(Y)

    # multiply a m x n matrix with an n x k matrix to get a m x k matrix,
    # where m = num examples, n = features, k = categories
    # result is each column of each row is the weight vector of that category
    # dot product(ed) with each example
    scores = np.dot(X, W)

    # perform softmax to get probabilities
    # m x k matrix
    probs = softmax(scores)

    # calculations for loss
    loss = (-1 / m) * np.sum(one_hot_Y * np.log(probs) + lam / 2 * np.sum(W*W))

    # n x k matrix
    grad = (-1 / m) * np.dot(x.T, (one_hot_Y - probs)) + lam * W

    return (loss, grad)

def softmax(z):
    """
    takes in a m x k matrix of weighted products,
    outputs an m x k matrix with the ith jth entry being
    the probability that example i is in category j
    """
    print z
    z -= np.max(z)
    probs = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    print probs
    return probs
    print "original: " + str(z)
    print np.sum(np.exp(z), axis=1)
    print np.exp(z).T / np.sum(np.exp(z), axis=1)
    # flattens the matrix, finds the max value
    # and then subtracts the max from every entry in z
    # !!! NOT SURE IF THIS IS NECESSARY??? !!!
    # possibly to prevent e^x from being too large
    z -= np.max(z)

    softmax = (np.exp(z) / np.sum(np.exp(z),axis=1)).T
    print sm
    print z

def getAccuracy(X, Y):
    """
    outputs the accuracy of the model for a given X and Y
    (total correct / total examples)
    """
    _, prediction = getProbsAndPreds(X)
    accuracy = sum(prediction == Y)/(float(len(Y)))
    return accuracy

mnist = mnist_data.read_data_sets("MNIST_data/", one_hot=False)
batch = mnist.train.next_batch(500)
tb = mnist.train.next_batch(100)

Y = batch[1]
X = batch[0]
testY = tb[1]
testX = tb[0]

print X
