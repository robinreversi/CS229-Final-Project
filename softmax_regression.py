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
    '''

    :param W: weight matrix where each entry column represents the kth category's
        weight for that feature
    :param X: data, m x n matrix
    :param Y: m x 1 vector, with each entry between 0 and k, to later be converted
        into a one-hot matrix
    :param lam: regularization parameter
    :return: the loss and gradient of training (currently batch GD)
    '''

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


# converts an m x 1 column vector with values in [0, k-1] to the corresponding one-hot m x k matrix
def makeOneHot(Y, k):
    '''
    Converts a vector of classes into the corresponding one-hot matrix

    :param Y: (m x 1) vector of classes corresponding to each input
    :param k: number of classes
    :return: (m x k) one-hot matrix from Y
    '''

    # number of rows
    m = Y.shape[0]
    OH = np.zeros(m, k)

    # put a 1 in the corresponding column for each row
    # e.g. if the class for example j is 5, put a 1 at (j, 5)
    for row in range(m):
        OH[row, Y[row]] = 1

    return OH


def softmax(z):
    '''

    :param z: m x k matrix of weighted products
    :return: m x k matrix with the ith jth entry being
    the probability that example i is in category j
    '''

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

# softmax(np.arange(6).reshape(3, 2))


def getPredictions(X, W):
    '''

    :param X: m x n data input
    :param W: n x k weights matrix
    :return: vector of probabilities of input belonging to classes,
            and vector of predicted class
    '''

    probabilities = softmax(np.dot(X, W))
    predictions = np.argmax(probabilities, axis=1)

    return probabilities, predictions


def softmaxRegression(X, Y, k):
    '''

    :param X: m x n data input
    :param Y: m x k one-hot matrix of classes
    :param k: number of classes
    :return: n x k weights matrix W, with columns the weight vectors
            corresponding to each class
    '''
    
    n = X.shape[1]
    W = np.zeros(n, k)

    # parameters that can be altered
    scale, iterations, learnRate = 1, 1000, 1e-5

    # loss vector is intended for plotting loss - not crucial
    # lossVec = []

    # batch gradient descent for given number of iterations
    for _ in range(iterations):
        loss, grad = getLoss(W, X, Y, scale)
        # lossVec.append(loss)
        W = W - learnRate * grad

    return W

