#
# This program runs softmax regression on a dataset
#
# Authors: Vince, Alex, Robin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from random import shuffle

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

    return probabilities, predictions


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
    scale, iterations, learnRate = 1, 1000, 1e-5

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
    _, prediction = getPredictions(X, W)
    accuracy = sum(prediction.reshape(Y.shape) == Y) / (float(len(Y)))
    return accuracy

data = pd.read_csv('train_data.csv').sample(frac=1)

X = np.array(data.iloc[:, 1:])
Y = np.array(data['0'].values).reshape((X.shape[0], 1))
p = np.random.permutation(X.shape[0])
shuffleX = X[p, :]
shuffleY = Y[p, :]

test_data = pd.read_csv('test_data.csv').sample(frac=1)
testX = test_data.iloc[:, 1:]
testY = test_data['0'].values

train_errors = []
dev_errors = []
'''
for i in range(50, X.shape[0], 10):
    print 'ITERS: ' + str(i)
    train_loss, dev_loss = softmaxRegression(shuffleX[0:i, :], shuffleY[0:i, :], 12, testX, testY)
    print train_loss
    train_errors.append(train_loss)
    dev_errors.append(dev_loss)

print train_errors
print dev_errors
'''
train_errors = [12.04028479228354, 13.45751774010404, 14.483991789149266, 15.926975111964301, 18.571494160266308, 21.558350699492081, 26.492528319559717, 28.297392434902985, 29.003319944107847, 30.284034530440017, 31.197416034111566, 33.193805293677414, 35.321229358369521, 37.759781912433894, 42.126089627009101, 44.21112806893116, 48.533280070040369, 51.487038476316854, 53.311675502522981, 57.233503525659259, 58.707834993377077, 60.751164059076913, 62.64582122047662, 65.671786266512072, 72.937262764908013, 74.622094414058779]
dev_errors = [395.46921040859934, 384.59031666860449, 390.18434701470562, 375.10567912362421, 370.91050408259741, 352.96964179284032, 338.60253095903226, 320.55602876640449, 317.02776221603699, 318.7810633160683, 321.47258149525248, 323.78691833953133, 314.16037962703058, 304.31386667264155, 295.69887317869836, 291.0993419803176, 282.94531523723981, 280.2773634545847, 272.43643991666625, 279.91608008636274, 280.74994355103462, 276.70330786676038, 275.10842417537378, 271.92706515066629, 272.50908413748982, 268.83977342129896]
plt.plot(range(50, X.shape[0], 10), train_errors)
plt.plot(range(50, X.shape[0], 10), dev_errors)
plt.show()

