import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=100000)


def getArtists():
    with open('./data_scraping/artists.txt') as f:
        return f.read().splitlines()

def nb_train(matrix, category, num_artists):
    state = {}
    N = matrix.shape[1]
    ###################

    for i in range(num_artists):
        key = "log-freq-" + str(i)

        # get all of the artist's songs
        artist_songs = matrix[np.where(category == i)]

        p_artist = len(artist_songs) / float(len(category))

        # get the total number of words across all the songs
        total_words = artist_songs.sum()

        # sum down the columns to get the number
        # of times a word appears across all spam
        # / not spam emails
        word_frequency = artist_songs.sum(axis=0)

        # calculate the probability of a word appearing 
        # in an email given that it's spam / not spam
        # and apply Laplace Smoothing
        phi_artist = np.multiply((1.0 / (total_words + N)), (word_frequency + 1))

        state[i] = p_artist
        state[key] = np.log(phi_artist)
    ###################
    return state

def nb_test(matrix, state, num_artists):
    output = np.zeros(matrix.shape[0])

    ###################
    # multiply the number of times each word in the email appears times 
    # its log probability that the word appears in a spam / not spam email
    # to get the relative probability that the email is spam / not spam
    estimates = []
    for i in range(num_artists):
        key = "log-freq-" + str(i)
        p_artist_i = matrix.dot(state[key]) + np.log(state[i])
        estimates.append(p_artist_i)
    total_matrix = np.array(estimates).T
    print total_matrix.shape

    output = np.argmax(total_matrix, axis=1)
    print output
    ###################
    return output

def evaluate(output, label):
    print
    print "OUTPUT: " + str(output)
    print label
    error = (output != label).sum() * 1. / len(output)
    print 'Accuracy: %1.4f' % (1 - error)
    return error

def findIndicators(state, tokenlist):
    # take the difference between the logs 
    # to get the relative probability
    log_diff = state['log_phi_spam'] - state['log_phi_nspam']
    # sort the list in descending order
    descending = np.argsort(log_diff)[::-1]
    # get the top 5 values
    top5 = descending[0:5]
    print np.array(tokenlist)[top5]


def main():
    trainMatrix = pd.read_csv('train_data.csv').sample(frac=1)
    testMatrix = pd.read_csv('test_data.csv').sample(frac=1)

    artists = getArtists()
    num_artists = len(artists)

    trainCategory = np.array(trainMatrix.iloc[:, 1])
    trainData = np.array(trainMatrix.iloc[:, 2:])

    state = nb_train(trainData, trainCategory, num_artists)

    print testMatrix

    testCategory = testMatrix.iloc[:, 1]
    print "TEST CATEOG: " + str(testCategory)
    testData = testMatrix.iloc[:, 2:]

#    output_train = nb_test(trainData, state, num_artists)

    output = nb_test(testData, state, num_artists)

    error = evaluate(output, testCategory)

#    evaluate(output_train, trainCategory)

    #findIndicators(state, )
    return

if __name__ == '__main__':
    main()
