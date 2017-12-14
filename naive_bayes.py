import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from operator import itemgetter
import sys

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

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

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

    output = np.argmax(total_matrix, axis=1)
    top_2 = np.argsort(total_matrix, axis=1)[:,[-1,-2]]
    ###################
    return output, top_2

def evaluate(output, label, artists, verbose=False):
    print "__EVALUATION__"
    error = (output != label).sum() * 1. / len(output)
    print 'Overall Accuracy: %1.4f' % (1 - error)
    if verbose:
        num_artists = len(artists)
        acc_per_artists = []
        for i in range(num_artists):
            artist_locs = np.where(label == i)
            artist_error = (output[artist_locs] != label[artist_locs]).sum() * 1. / len(artist_locs[0])
            acc_per_artists.append((artists[i], 1 - artist_error))

        acc_per_artists.sort(key=itemgetter(1), reverse=True)
        for artist, value in acc_per_artists:
            print "Artist: " + str(artist)
            print "Accuracy: " + str(value)
            print

    return error

def evaluateTop2(top_2, label, artists):
    print("----------------------------")
    print("TOP 2")
    error = ((1 -np.any([top_2[:, 0] == label, top_2[:, 1] == label], axis=0)).sum()) * 1. / float(top_2.shape[0])
    print 'Overall Accuracy: %1.4f' % (1 - error)
    num_artists = len(artists)
    #print len(label)
    acc_per_artists = []
    for i in range(num_artists):
        artist_locs = np.where(label == i)
        #print top_2[artist_locs, 0]
        artist_accuracy = np.any([top_2[artist_locs, 0] == label[artist_locs],
                        top_2[artist_locs, 1] == label[artist_locs]], axis=0).sum() * 1. / len(artist_locs[0])
        acc_per_artists.append((artists[i], artist_accuracy))
        print "NUM GUESSED FOR " + str(artists[i])

    acc_per_artists.sort(key=itemgetter(1), reverse=True)
    for artist, value in acc_per_artists:
        print "Artist: " + str(artist)
        print "Accuracy: " + str(value)
        print

    print("---------------------------")
    return error

def findIndicators(state, tokenlist):
    # take the difference between the logs 
    # to get the relative probability
    log_diff = state['log_phi_spam'] - state['log_phi_nspam']
    # sort the list in descending order
    descending = np.argsort(log_diff)[::-1]
    # get the top 5 values
    top5 = descending[0:5]


def main():
    if len(sys.argv) > 1:
        lower = sys.argv[1]
        upper = sys.argv[2]
        TF = sys.argv[3]

        strain = 'train_' + str(lower) + '-' + str(upper) + '_' + TF + '.csv'
        sdev = 'dev_' + str(lower) + '-' + str(upper) + '_' + TF + '.csv'
        stest = 'test_' + str(lower) + '-' + str(upper) + '_' + TF + '.csv'
        
        trainMatrix = pd.read_csv(strain)#.sample(frac=1)
        devMatrix = pd.read_csv(sdev)#.sample(frac=1)
        testMatrix = pd.read_csv(stest)#.sample(frac=1)

    else:
        trainMatrix = pd.read_csv('train_10-1000_binary.csv').sample(frac=1)
        testMatrix = pd.read_csv('dev_10-1000_binary.csv').sample(frac=1)

    artists = getArtists()
    num_artists = len(artists)

    trainCategory = np.array(trainMatrix.iloc[:, 0])
    trainData = np.array(trainMatrix.iloc[:, 2:])

    devCategory = np.array(devMatrix.iloc[:, 0])
    devData = np.array(devMatrix.iloc[:, 2:])

    testCategory = np.array(testMatrix.iloc[:, 0])
    testData = np.array(testMatrix.iloc[:, 2:])

    state = nb_train(trainData, trainCategory, num_artists)
    
    print("TRAIN Accuracy")
    train_output = nb_test(trainData, state, num_artists)[0]
    train_error = evaluate(train_output, trainCategory, artists)

    print("DEV Accuracy")
    dev_output = nb_test(devData, state, num_artists)[0]
    dev_error = evaluate(dev_output, devCategory, artists)

    print("TEST Accuracy: ")
    output, _ = nb_test(testData, state, num_artists)
    error = evaluate(output, testCategory, artists)
    #evaluateTop2(top_2, testCategory, artists)

    

    #findIndicators(state, )

if __name__ == '__main__':
    main()
