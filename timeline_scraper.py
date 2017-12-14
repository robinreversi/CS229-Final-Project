import pandas as pd
import kanye_periods
#from py_genius import Genius
from vocabulary_builder import buildVocabulary
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer as ps
import numpy as np
import sys
ARTIST = "Kanye West"
ps = ps()

'''
# Given a song 'title' returns the period it belongs to
def set_period(title):
    print("Song: ", title)
    result = gen.search(title)['response']['hits']

    for i in range(len(result)):
        if(result[i]['result']['primary_artist']['name'] == ARTIST):
            album = gen.get_song(result[i]['result']['id'])['response']['song']['album']
            album = album if not album else album['name']
            if(album in kanye_periods.PERIOD_1):
                return 0
            elif(album in kanye_periods.PERIOD_2):
                return 1
            elif(album in kanye_periods.PERIOD_3):
                return 2
            else:
                return None



raw = pd.read_csv('data_scraping/finaldata.csv', delimiter="|")
artist_data = raw[raw['Artist'] == ARTIST]
gen = Genius('thRsqBz-IHPVCj0IU6_VNpRjRs9o2HGNxN_WAAoXCyOJPuRbhyb0MSJGNFcTnnlQ')

artist_data['Artist'] = artist_data['Title'].apply(set_period)
artist_data = artist_data.dropna(subset = ['Artist'])
artist_data.to_csv(ARTIST + '_t_data.csv', sep="|", index=False)
'''

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


def split_data(data, test_size):
    train_data = pd.DataFrame(columns=['Artist', 'Title', 'Lyrics'])
    test_data = pd.DataFrame(columns=['Artist', 'Title', 'Lyrics'])
    for i in range(len(kanye_periods.PERIODS)):
        artist_data = data[data['Artist'] == i]
        artist_train, artist_test = train_test_split(artist_data, test_size=test_size, random_state=420)
        train_data = train_data.append(artist_train)
        test_data = test_data.append(artist_test)
    return train_data, test_data

def featureExtractor(raw_data, filename, vocab, lower=0, upper=20000, TF='regular', verbose=0):
    # setup
    # set of all words that appear in the song
    processed_data = []
    #TF = 'binary'
    K = 0.5

    for data_pt in raw_data:
        label = data_pt[0]
        print label
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

        phi = ([1] + list(vocab_dict.values()))
        processed_data.append([label] + phi)


    if(TF != 'binary'):
        processed_data = np.array(processed_data)
        x = processed_data[:, 2:]
        y = processed_data[:, 0:2]
        x = normalize(x)
        processed_data = np.append(y, x, axis=1)

    processed_df = pd.DataFrame(processed_data)
    print(processed_df.head())
    processed_df.to_csv(filename + '_' + TF + '.csv', index=False)

def initialize_data_sets(raw_data_path):
    data = pd.read_csv(raw_data_path, delimiter='|')
    train_data, test_data = split_data(data, .15)
    train_data, dev_data = split_data(train_data, .125)

    pd.DataFrame(train_data, columns=["Artist", "Title", "Lyrics"]).to_csv('t_kanye_train.csv', sep="|", index=False)
    pd.DataFrame(dev_data, columns=["Artist", "Title", "Lyrics"]).to_csv('t_kanye_dev.csv', sep="|", index=False)
    pd.DataFrame(test_data, columns=["Artist", "Title", "Lyrics"]).to_csv('t_kanye_test.csv', sep="|", index=False)


def main():
    initialize_data_sets('Kanye West_t_data.csv')
    train_data = pd.read_csv('t_kanye_train.csv', delimiter='|').as_matrix()
    dev_data = pd.read_csv('t_kanye_dev.csv', delimiter='|').as_matrix()
    test_data = pd.read_csv('t_kanye_test.csv', delimiter='|').as_matrix()

    if len(sys.argv) > 1:
        lower = sys.argv[1]
        upper = sys.argv[2]
        TF = sys.argv[3] if len(sys.argv) > 3 else 'regular'

        strain = 'kanye_train_' + str(lower) + '-' + str(upper)
        sdev = 'kanye_dev_' + str(lower) + '-' + str(upper)
        stest = 'kanye_test_' + str(lower) + '-' + str(upper)
        vocab = buildVocabulary(10, 1000, 't_kanye_train.csv')
        featureExtractor(train_data, strain, vocab, lower, upper, TF)
        featureExtractor(dev_data, sdev, vocab, lower, upper, TF, vocab)
        featureExtractor(test_data, stest, vocab, lower, upper, TF, vocab)

if __name__ == "__main__":
    main()