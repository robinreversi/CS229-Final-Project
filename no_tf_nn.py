import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd


def readData(train_file, dev_file, test_file):
    train = pd.read_csv(train_file)
    print train.head()
    dev = pd.read_csv(dev_file)
    test = pd.read_csv(test_file)
    return np.array(train), np.array(dev), np.array(test)

def softmax(x):
    """
    Compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    """
    ### YOUR CODE HERE
    maxs = np.amax(x, axis=1)
    x = np.subtract(x.T, maxs).T
    x_exp = np.exp(x)
    s = (x_exp.T / np.sum(x_exp, axis=1)).T
    ### END YOUR CODE
    return s

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    s = 1.0 / (1  + np.exp(-x))
    return s

def L(y, labels, params=None, reg=False):
    reg_term = .0001
    loss = cross_entropy(y, labels)
    if(reg):
        W1 = params['W1']
        W2 = params['W2']
        loss += reg_term * (np.linalg.norm(W1) ** 2 + np.linalg.norm(W2) ** 2)
    return loss


def cross_entropy(y, labels):
    (m, k) = y.shape
    log_y = np.log(y)
    product = np.multiply(log_y, labels)
    loss = -1 * np.sum(product) / float(m)
    return loss

def forward_prop(data, labels, params, reg):
    """
    return hidden layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    z_1 = data.dot(W1) + b1
    h = sigmoid(z_1)
    z_2 = h.dot(W2) + b2
    y = softmax(z_2)
    print "AFTER SOFTMAX: " + str(y.shape)
    cost = L(y, labels, params, reg)
    ### END YOUR CODE
    return h, y, cost

def backward_prop(data, labels, params, h, y, reg_term=.0001, reg=False):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    (m, k) = y.shape
    print "Y SHAPE:" + str(y.shape)
    ### YOUR CODE HERE
    delta_2 = 1.0 / m * (y - labels)
    gradW2 = h.T.dot(delta_2) 
    if(reg):
        gradW2 += reg_term * 2 * W2
    gradb2 = np.sum(delta_2, axis=0).T
    delta_1 = delta_2.dot(W2.T) * (h * (1-h))
    gradW1 = data.T.dot(delta_1)
    if(reg):
        gradW1 += reg_term * 2 * W1
    gradb1 = np.sum(delta_1, axis=0).T
    ### END YOUR CODE

    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2

    #print grad

    return grad

def nn_train(trainData, trainLabels, devData, devLabels, reg):
    print "Training"
    (m, n) = trainData.shape
    num_hidden = 300
    learning_rate = 5
    params = {}
    (m, k) = trainLabels.shape

    ### YOUR CODE HERE
    # initialize weights to random values sampled from a normal
    params['W1'] = np.random.normal(0, 1, (n, num_hidden))
    params['b1'] = np.zeros(num_hidden)
    params['W2'] = np.random.normal(0, 1, (num_hidden, k))
    params['b2'] = np.zeros(k)

    train_loss = []
    dev_loss = []

    train_accuracy = []
    dev_accuracy = []

    # for 30 epochs, perform fwd / backward propogation
    for i in range(30):
        # one epoch = 50 batch iterations
        print i
        for j in range(50):
            start = j * 1000
            end = (j + 1) * 1000
            batch_data = trainData[start:end, :]
            batch_labels = trainLabels[start:end, :]
            # for each training batch do fwd / bkwd propogation and update params
            (h, y, cost) = forward_prop(batch_data, batch_labels, params, reg)
            grad = 0
            if(reg):
                grad = backward_prop(batch_data, batch_labels, params, h, y, .0001, True)
            else:
                grad = backward_prop(batch_data, batch_labels, params, h, y)
            params['W1'] -= learning_rate * grad['W1']
            params['W2'] -= learning_rate * grad['W2']
            params['b1'] -= learning_rate * grad['b1']
            params['b2'] -= learning_rate * grad['b2']

        (_, train_y, train_cost) = forward_prop(trainData, trainLabels, params, False)
        (_, dev_y, dev_cost) = forward_prop(devData, devLabels, params, False)

        # at the end of each epoch, calculate loss (one plot) and accuracy (another plot)
        # on entire train set and dev set and store for plotting later

        train_loss.append(train_cost)
        dev_loss.append(dev_cost)

        train_accuracy.append(compute_accuracy(train_y, trainLabels))
        dev_accuracy.append(compute_accuracy(dev_y, devLabels))


    plt.figure(1)
    plt.subplot(211)
    plt.plot(range(30), train_loss, 'r', label='Train')
    plt.plot(range(30), dev_loss, 'b', label='Dev')
    plt.legend()

    plt.title('Loss vs. Epochs')

    plt.subplot(212)
    plt.plot(range(30), train_accuracy, 'r', label='Train')
    plt.plot(range(30), dev_accuracy, 'b', label='Dev')
    plt.title('Accuracy vs. Epochs')
    plt.legend(loc="lower right")
    plt.show()

    # save learnt parameters



    output = open('reg_' + str(reg) + '_weights.pkl', 'wb')
    pickle.dump(params, output)

    ### END YOUR CODE

    return params


def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params, False)
    accuracy = compute_accuracy(output, labels)
    return accuracy


def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy


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


def main():
    np.random.seed(100)
    train, dev, test = readData('train_10-1000_norm.csv', 'dev_10-1000_norm.csv', 'test_10-1000_norm.csv')
    trainData = train[:, 1:]
    trainLabels = makeOneHot(train[:, 0].astype(int), 12)
    p = np.random.permutation(trainData.shape[0])
    trainData = trainData[p,:]
    trainLabels = trainLabels[p,:]

    devData = dev[:, 1:]
    devLabels = dev[:, 0]
    
    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testLabels = makeOneHot(test[:, 0].astype(int), 12)
    testData = (test[:, 1:] - mean) / std

    params = nn_train(trainData, trainLabels, devData, devLabels, True)

    readyForTesting = True
    if readyForTesting:
        accuracy = nn_test(testData, testLabels, params)
        print 'Test accuracy: %f' % accuracy

if __name__ == '__main__':
    main()
