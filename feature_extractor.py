 
# Feature extractor for CS 229 Final Project: Whose Rap is it Anyways?
# Alex Wang, Robin Cheong, Vince Ranganathan
# jwang98, robinc20, akranga @ stanford.edu
# Updated 11/02/2017

from nltk.stem import PorterStemmer as ps
from vocabulary_builder import buildVocabulary
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import numpy as np
import regex as re

ps = ps()
VOCAB_FILE = "chosen_train.csv"

#----------------------------------#

def normalize(data):
    bias = data[:, 0]
    m, n = data.shape
    means = (data.sum(axis=0) * 1.0 / m)
    data = data - means
    variance = np.square(data).sum(axis=0) * 1.0 / m
    variance[variance == 0] = 1
    data /= np.sqrt(variance)    
    data[:, 0] = bias
    return data


def getArtists():
    with open('./data_scraping/artists.txt') as f:
        return f.read().splitlines()


def extract_artist_map():
    artists = getArtists()
    artist_map = {}
    for i, artist in enumerate(artists):
        artist_map[artist] = i
    return artist_map

def calc_idf(data, vocab):
    m, n = data.shape

    appearances = []
    for i, data_pt in enumerate(data):
        vocab_dict = dict.fromkeys(vocab, 0)
        words = data_pt[2].decode('utf-8').split()
        lyrics = [ps.stem(word) for word in words]

        for word in lyrics:
            word = re.sub(r'\p{P}+', "", word)
            if word in vocab_dict:
                vocab_dict[word] = 1

        counts = np.array(vocab_dict.values())
        appearances.append(counts)
    appearances = np.array(appearances).sum(axis=0)
    idf = np.log(float(m) / appearances)
    return idf


def featureExtractor(raw_data, filename, vocab, idf, lower=0, upper=20000, TF='regular', verbose=0):
    # setup
    # set of all words that appear in the song
    processed_data = []
    artist_map = extract_artist_map()
    K = 0.5

    for data_pt in raw_data:
        artist = artist_map[data_pt[0]]

        vocab_dict = dict.fromkeys(vocab, 0)
        words = data_pt[2].decode('utf-8').split()
        lyrics = [ps.stem(word) for word in words]

        def wordFrequencies(vocab_dict, lyrics):
            for word in lyrics:
                if word in vocab_dict:
                    vocab_dict[word] += 1

        if TF == 'binary':
            for word in lyrics:
                if word in vocab_dict:
                    vocab_dict[word] = 1

        elif TF == 'regular':
            wordFrequencies(vocab_dict, lyrics)

        elif TF == 'log':
            wordFrequencies(vocab_dict, lyrics)
            for word in lyrics:
                if word in vocab_dict:
                    vocab_dict[word] = np.log(1 + vocab_dict[word])

        elif TF == 'norm':
            wordFrequencies(vocab_dict, lyrics)
            max_freq = max(vocab_dict.values())
            for word in lyrics:
                if word in vocab_dict:
                    vocab_dict[word] = K + ((1 - K) * (vocab_dict[word] / max_freq))

        counts = np.array(vocab_dict.values())
        phi = np.concatenate((np.array([1]), counts))
        processed_data.append(np.concatenate((np.array([artist]), phi)))

    if(TF != 'binary'):
        processed_data = np.array(processed_data)
        x = processed_data[:, 2:] * idf
        y = processed_data[:, 0:2]
        processed_data = np.append(y, x, axis=1)


    processed_df = pd.DataFrame(processed_data)
    print processed_df.head()
    processed_df.to_csv(filename + '_' + TF + '.csv', index=False)


#----------------------------------#

def split_data(data, test_size):
    artists = getArtists()
    train_data = pd.DataFrame(columns=['Artist', 'Title', 'Lyrics'])
    test_data = pd.DataFrame(columns=['Artist', 'Title', 'Lyrics'])
    for artist in artists:
        artist_data = data[data['Artist'] == artist]
        artist_train, artist_test = train_test_split(artist_data, test_size=test_size, random_state=420)
        train_data = train_data.append(artist_train)
        test_data = test_data.append(artist_test)
    return train_data, test_data

def initialize_data_sets(raw_data_path):
    data = pd.read_csv(raw_data_path, delimiter='|')
    train_data, test_data = split_data(data, .15)
    train_data, dev_data = split_data(train_data, .125)

    pd.DataFrame(train_data, columns=["Artist", "Title", "Lyrics"]).to_csv('chosen_train.csv', sep="|", index=False)
    pd.DataFrame(dev_data, columns=["Artist", "Title", "Lyrics"]).to_csv('chosen_dev.csv', sep="|", index=False)
    pd.DataFrame(test_data, columns=["Artist", "Title", "Lyrics"]).to_csv('chosen_test.csv', sep="|", index=False)

def main():
    #initialize_data_sets('./data_scraping/finaldata.csv')

    train_data = pd.read_csv('chosen_train.csv', delimiter='|').as_matrix()
    dev_data = pd.read_csv('chosen_dev.csv', delimiter='|').as_matrix()
    test_data = pd.read_csv('chosen_test.csv', delimiter='|').as_matrix()

    if len(sys.argv) > 1:
        lower = sys.argv[1]
        upper = sys.argv[2]
        TF = sys.argv[3] if len(sys.argv) > 3 else 'regular'

        strain = 'train_' + str(lower) + '-' + str(upper)
        sdev = 'dev_' + str(lower) + '-' + str(upper)
        stest = 'test_' + str(lower) + '-' + str(upper)
        vocab = buildVocabulary(lower, upper, VOCAB_FILE)
        idf = calc_idf(train_data, vocab)
        featureExtractor(train_data, strain, vocab, idf, lower, upper, TF)
        featureExtractor(dev_data, sdev, vocab, idf, lower, upper, TF, vocab)
        featureExtractor(test_data, stest, vocab, idf, lower, upper, TF, vocab)

if __name__ == "__main__":
    main()
