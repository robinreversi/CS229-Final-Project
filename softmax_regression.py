# This program runs softmax regression on a dataset
#
# Authors: Vince, Alex, Robin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from random import shuffle
import sys

def getLoss(W, X, Y, lam):
    '''

    :param W: n x k weight matrix where each entry column represents the kth category's
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
    k = W.shape[1]
    one_hot_Y = makeOneHot(Y, k)

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
    grad = (-1 / m) * np.dot(X.T, (one_hot_Y - probs)) + lam * W

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
    OH = np.zeros((m, k))

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
    z -= np.max(z)
    probs = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return probs
    
def getPredictions(X, W):
    '''

    :param X: m x n data input
    :param W: n x k weights matrix
    :return: vector of probabilities of input belonging to classes,
            and vector of predicted class
    '''

    probabilities = softmax(np.dot(X, W))
    predictions = np.argmax(probabilities, axis=1)
    top_2 = np.argsort(probabilities, axis=1)[:,[-1,-2]]


    return probabilities, predictions, top_2


def softmaxRegression(X, Y, k, testX, testY):
    '''

    :param X: m x n data input
    :param Y: m x k one-hot matrix of classes
    :param k: number of classes
    :return: n x k weights matrix W, with columns the weight vectors
            corresponding to each class
    '''
    n = X.shape[1]
    W = np.zeros((n, k))

    # parameters that can be altered
    scale, iterations, learnRate = 3, 1000, 1e-5

    # loss vector is intended for plotting loss - not crucial
    lossVec = []

    # batch gradient descent for given number of iterations
    for _ in range(iterations):
        loss, grad = getLoss(W, X, Y, scale)
        lossVec.append(loss)
        W = W - learnRate * grad
    #plt.plot(lossVec)
    print "Train Accuracy: " + str(getAccuracy(X, Y, W))
    print "Test Accuracy: " + str(getAccuracy(testX, testY, W))
    #plt.show()

    train_loss = getLoss(W, X, Y, scale)[0]
    dev_loss = getLoss(W, testX, testY, scale)[0]
    return train_loss, dev_loss


def getAccuracy(X, Y, W):
    """
    outputs the accuracy of the model for a given X and Y
    (total correct / total examples)
    """
    _, prediction, top_2 = getPredictions(X, W)
    accuracy = sum(prediction.reshape(Y.shape) == Y) / (float(len(Y)))
    print accuracy
    #mistakes = np.where(prediction.reshape(Y.shape) != Y)[0]
    #print mistakes
    #print "MISTAKES MADE"
    #for mistake in mistakes:
    #    print mistake
    #    print "PREDICTION: " + "[" + str(prediction[mistake]) + "]"
    #    print "ACTUAL: " + str(Y[mistake]) 
    #    print

    top2_acc = np.any([top_2[:, 0] == Y[:, 0], top_2[:, 1] == Y[:, 0]], axis=0).sum() * 1. / len(Y)
    print "TOP2"
    print top2_acc
    return accuracy

if len(sys.argv) > 1:
    lower = sys.argv[1]
    upper = sys.argv[2]

    strain = 'train_' + str(lower) + '-' + str(upper) + '.csv'
    stest = 'test_' + str(lower) + '-' + str(upper) + '.csv'

    data = pd.read_csv(strain).sample(frac=1)
    test_data = pd.read_csv(stest).sample(frac=1)
else:
    data = pd.read_csv('normalized_train_3-2000.csv').sample(frac=1)
    test_data = pd.read_csv('normalized_test_3-2000.csv').sample(frac=1)

X = np.array(data.iloc[:, 1:])
Y = np.array(data['0'].values).reshape((X.shape[0], 1)).astype(int)


testX = np.array(test_data.iloc[:, 1:])
testY = np.array(test_data['0'].values).reshape((testX.shape[0], 1)).astype(int)

train_loss, dev_loss = softmaxRegression(X, Y, 12, testX, testY)
'''
train_errors = []
dev_errors = []

for i in range(50, X.shape[0], 10):
    print 'ITERS: ' + str(i)
    train_loss, dev_loss = softmaxRegression(X[0:i, :], Y[0:i, :], 12, testX, testY)
    print train_loss
    train_errors.append(train_loss)
    dev_errors.append(dev_loss)

print train_errors
print dev_errors

plt.plot(range(50, X.shape[0], 10), train_errors)
plt.plot(range(50, X.shape[0], 10), dev_errors)
plt.show()

'''