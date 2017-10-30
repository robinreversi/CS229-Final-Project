#
# This program runs softmax regression on a dataset
#
# Authors: Vince, Alex, Robin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import scipy.sparse


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
    prob = softmax(scores)

    # calculations for loss
    loss = (-1 / m) * np.sum(one_hot_Y * np.log(prob) + lam / 2 * np.sum(W*W))

    # n x k matrix
    grad = (-1 / m) * np.dot(x.T, (one_hot_Y - prob)) + lam * W

    return (loss, grad)

def softmax(z):
    z -= np.max(z)
