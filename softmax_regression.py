# This program runs softmax regression on a dataset
#
# Authors: Vince, Alex, Robin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from random import shuffle
import sys
from vocabulary_builder import buildVocabulary
import random

def getArtists():
    with open('./data_scraping/artists.txt') as f:
        return f.read().splitlines()

def getLoss(W, X, Y, lamb):
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
    loss = (-1 / m) * np.sum(one_hot_Y * np.log(probs)) + lamb / 2.0 * np.linalg.norm(W)
    # n x k matrix
    grad = (-1 / m) * np.dot(X.T, (one_hot_Y - probs)) + lamb * W
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

def softmaxRegression(train_x, train_y, dev_x, dev_y, test_x, test_y, max_iters=300, num_classes=12, lamb=1):
    '''

    :param X: m x n data input
    :param Y: m x k one-hot matrix of classes
    :param k: number of classes
    :return: n x k weights matrix W, with columns the weight vectors
            corresponding to each class
    '''
    m, n = train_x.shape
    W = np.zeros((n, num_classes))

    # parameters that can be altered
    learnRate = 1e-4
    BATCH_SIZE = int(m / 1.0)
    NUM_BATCHES = int(m / float(BATCH_SIZE))
    EPSILON = 1e-5  
    # loss vector is intended for plotting loss - not crucial
    lossVec = []
    iters = 0
    # batch gradient descent for given number of iterations
    prev_loss = np.inf
    orig_x = train_x
    orig_y = train_y
    while(True):
    #for _ in range(max_iters):
        prevW = W
        p = np.random.permutation(train_x.shape[0])
        train_x = train_x[p, :]
        train_y = train_y[p, :]
        for i in range(NUM_BATCHES):
            start = i * BATCH_SIZE
            end = (i + 1) * BATCH_SIZE
            loss, grad = getLoss(W, train_x[start:end, :], train_y[start:end, :], lamb)
            lossVec.append(loss)
            W = W - learnRate * grad
        norm = np.linalg.norm(prevW - W)
        if(norm < EPSILON or iters > max_iters):
            break
        if(norm < 1e-2):
            learnRate = norm / (10.0)
        iters += 1
        print "Norm: " + str(norm) + "\tIter: " + str(iters) + "\tLoss: " + str(loss)
  



    print W
    #plt.plot(lossVec)  

    train_acc = getAccuracy(orig_x, orig_y, W)
    dev_acc = getAccuracy(dev_x, dev_y, W)
    test_acc = getAccuracy(test_x, test_y, W)
    print("Train Accuracy: ", train_acc)
    print("Dev Accuracy: ", dev_acc)
    print("Test Accuracy: ", test_acc)
    #plt.show()

    #analyze_features(W) 
    print lamb
    print max_iters
    train_loss = getLoss(W, train_x, train_y, lamb)[0]
    dev_loss = getLoss(W, dev_x, dev_y, lamb)[0]
    return W, train_loss, dev_loss, train_acc, dev_acc

def getAccuracy(X, Y, W):
    """
    outputs the accuracy of the model for a given X and Y
    (total correct / total examples)
    """
    _, prediction, top_2 = getPredictions(X, W)
    print "PREDICTIONS: "
    mistakes = np.array([[0 for _ in range(12)] for _ in range(12)])
    for i, label in enumerate(prediction):
        print 'Index: ' + str(i) + '\tLabel: ' + str(label) + '\tSecond Guess: ' + str(top_2[i][1]) + '\tTrue Val: ' + str(Y[i])
        if(label != Y[i]):
            mistakes[Y[i][0]][label] += 1
    total_mistakes = mistakes.sum(axis=1)
    for i, row in enumerate(mistakes):
        print "Total Misakes on artist: " + str(i) + '\t' + str(total_mistakes[i])
        print "Per artist: " + str(mistakes[i]) + "\n"

    accuracy = sum(prediction.reshape(Y.shape) == Y) / (float(len(Y)))
    top2_acc = np.any([top_2[:, 0] == Y[:, 0], top_2[:, 1] == Y[:, 0]], axis=0).sum() * 1. / len(Y)
    print "TOP2"
    print top2_acc
    return accuracy

def main():
    if len(sys.argv) > 1:
        lower = sys.argv[1]
        upper = sys.argv[2]
        TF = sys.argv[3]
        artist = sys.argv[4] + '_' if len(sys.argv) > 4 else ''

        strain = artist + 'train_' + str(lower) + '-' + str(upper) + '_' + TF + '.csv'
        sdev = artist + 'dev_' + str(lower) + '-' + str(upper) + '_' + TF + '.csv'
        stest = artist + 'test_' + str(lower) + '-' + str(upper) + '_' + TF + '.csv'

        train = pd.read_csv(strain)#.sample(frac=1)
        dev = pd.read_csv(sdev)#.sample(frac=1)
        test = pd.read_csv(stest)#.sample(frac=1)

    train_x = np.array(train.iloc[:, 1:])
    train_y = np.array(train['0'].values).reshape((train_x.shape[0], 1)).astype(int)

    dev_x = np.array(dev.iloc[:, 1:])
    dev_y = np.array(dev['0'].values).reshape((dev_x.shape[0], 1)).astype(int)

    #comb_x = np.append(train_x, dev_x, axis=0)
    #comb_y = np.append(train_y, dev_y, axis=0)

    test_x = np.array(test.iloc[:, 1:])
    test_y = np.array(test['0'].values).reshape((test_x.shape[0], 1)).astype(int)

    #print comb_x.shape
    softmaxRegression(train_x, train_y, dev_x, dev_y, test_x, test_y, lamb=0)
    #softmaxRegression(comb_x, comb_y, test_x, test_y, test_x, test_y)
    #plot_learning_curve(train_x, train_y, dev_x, dev_y, test_x, test_y)


def analyze_features(W):
    print("-----------------------------------------------")
    print("ANALYSIS OF MOST IMPORTANT FEATURES FOR ARTISTS")

    def indicative(W):
        imp = np.array(W)
        for k in range(imp.shape[0]):          # feature
            for i in range(imp.shape[1]):      # rapper
                imp[k, i] = imp.shape[1] * W[k, i] - sum(W[k, :])
        return imp

    most_important = np.argsort(indicative(W[1:, :]).T, axis=1)[:, -5:]

    # most_important = np.argsort(W.T, axis=1)[:, 0:5]
    artists = getArtists()
    print(most_important)
    vocab = np.array(list(buildVocabulary(10, 1000, 'chosen_train.csv')))
    for i, row in enumerate(most_important):
        print("MOST IMPORTANT FEATURES FOR ARTIST: " + str(artists[i]))
        print vocab[row]
    print("-----------------------------------------------")

def test_lambdas(train_x, train_y, dev_x, dev_y):
    lambdas = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    train_losses = []
    dev_losses = [] 
    train_accs = []
    dev_accs = [] 

    for lamb in lambdas:
        print("LAMB", lamb)
        train_loss, dev_loss, train_acc, dev_acc = softmaxRegression(train_x, train_y, dev_x, dev_y, lamb=lamb)
        train_losses.append(train_loss)
        dev_losses.append(dev_loss)
        train_accs.append(train_acc)
        dev_accs.append(dev_acc)

    print(train_losses)
    print(dev_losses)
    print(train_accs)
    print(dev_accs)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(lambdas, train_losses, 'r', label='Train')
    plt.plot(lambdas, dev_losses, 'b', label='Dev')
    plt.legend()

    plt.title('Loss vs. Lambdas')

    plt.subplot(212)
    plt.plot(lambdas, train_accs, 'r', label='Train')
    plt.plot(lambdas, dev_accs, 'b', label='Dev')
    plt.title('Accuracy vs. Lambas')
    plt.legend(loc="lower right")
    plt.show()

def plot_learning_curve(train_x, train_y, dev_x, dev_y, test_x, test_y):
    train_errors = []
    dev_errors = []

    train_sizes = range(50, train_x.shape[0], 150)

    for i in train_sizes:
        print 'ITERS: ' + str(i)
        W, train_loss, dev_loss, train_acc, dev_acc = softmaxRegression(train_x[0:i, :], train_y[0:i, :], dev_x, dev_y, test_x, test_y, lamb = 0)
        print train_loss
        train_errors.append(train_loss)
        dev_errors.append(dev_loss)

    print train_errors
    print dev_errors

    train = plt.plot(train_sizes, train_errors)
    dev = plt.plot(train_sizes, dev_errors)
    plt.ylabel('Loss')
    plt.xlabel('Training Size')
    plt.title('Loss v. Training Size')
    plt.legend(['Train', 'Dev'])#, handles=[train, dev])
    plt.show()

if __name__ == '__main__':
    main()
