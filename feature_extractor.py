 
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

ps = ps()

#----------------------------------#

def normalize(data):
    m, n = data.shape
    means = (data.sum(axis=0) * 1.0 / m)
    data = data - means
    variance = np.square(data).sum(axis=0) * 1.0 / m
    variance[variance == 0] = 1
    data /= np.sqrt(variance)    
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


artist_map = extract_artist_map()

def featureExtractor(raw_data, filename, vocab, lower=0, upper=20000, TF='regular', verbose=0):
    '''
    ---FEATURES 'KEYNAME' : DESCRIPTION---
    All feature keynames begin with an underscore (_). Others are words in the vocabulary.
    -Intercept '_incpt': 1

    --Vocabulary
    -Vocabulary richness '_vocabRich': (float) fraction of words that are unique
    -Vocabulary 'word': (int) number of occurrences of word
    -Abusive language distribution '_nAbuses%d': (int) number of lines with %d abuses

    --Length
    -Line length distribution '_linesLen%d': (int) number of lines of length %d words
    -Word length distribution '_wordsLen%d': (int) number of words with %d chars
    -Verse length '_verseLen': (int) length of the verse in lines

    --Rhyme
    -Weak rhyme approximation '_nWeakRhymes': (int) number of weak rhymes (up to 2 lines ahead, exactly last two chars match)
    -Strong rhyme approximation '_nStrongRhymes': (int) number of strong rhymes (up to 2 lines ahead, exactly last three chars match)

    :param input: a string of text (with lines delimited by '\n'?) representing a single verse
    :return: the feature vector for the input
    '''

    abusiveLang = {}  # TODO put in a list of bad words?
    """
    def initialize(phi, vocabulary):
        '''
        :param phi: as given
        :param verse: as given
        :return: n/a (phi initialized)
        '''
        # initializing the feature vector as a dict
        phi['_incpt'] = 1
        phi['_verseLen'] = len(verse)
    """
    #--------EXECUTE--------#

    def preprocessText(lyrics):
        '''
        :param input: verse as a single string, (delimited by '\n'?)
        :return: list where elements are lines of the verse
        '''
        words = lyrics.decode('utf-8').split()
        return [ps.stem(word) for word in words]

    # setup
    # set of all words that appear in the song
    processed_data = []

    #TF = 'binary'
    K = 0.5

    '''
    --- for IDF ---
    
    word_doc_counter = dict.fromkeys(vocab, 0)
    n_docs = 0.0
    
    for data_pt in raw_data:
        lyrics = preprocessText(data_pt[2])

        update_wdcounter = dict.fromkeys(vocab, 0)
        for word in lyrics:
            if word in update_wdcounter:
                update_wdcounter[word] = 1

        for word in word_doc_counter:
            word_doc_counter[word] += update_wdcounter[word]

        n_docs += 1.0
    '''

    for data_pt in raw_data:
        artist = artist_map[data_pt[0]]
        vocab_dict = dict.fromkeys(vocab, 0)
        lyrics = preprocessText(data_pt[2])

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

        '''
        IDF = True
        if IDF:
            for word in lyrics:
                if word in vocab_dict:
                    vocab_dict[word] *= np.log(n_docs / word_doc_counter[word])
        '''

        phi = ([1] + list(vocab_dict.values()))
        processed_data.append([artist] + phi)

    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(filename + '_' + TF + '.csv')


#----------------------------------#
'''
def test():
    input = "The world is spinning\n" \
            "The days are's changing\n" \
            "The lives are [rest] hungry\n" \
            "And I am so funny\n" \
            "I like to eat food\n" \
            "Does that rhyme with rod\n" \
            "Fucking hell dude\n" \
            "Hello all my friends"
    phi = featureExtractor(input, verbose=1)
    print(phi)

test()
'''
data = pd.read_csv('data_scraping/finaldata.csv', delimiter='|')
train_data, test_data = train_test_split(data,test_size=0.20, random_state=420)
train_data, dev_data = train_test_split(train_data, test_size=.25, random_state=420)
train_data = train_data.as_matrix()
dev_data = dev_data.as_matrix()
test_data = test_data.as_matrix()

pd.DataFrame(train_data, columns=["Artist", "Title", "Lyrics"]).to_csv('chosen_train.csv', sep="|")
pd.DataFrame(dev_data, columns=["Artist", "Title", "Lyrics"]).to_csv('chosen_dev.csv', sep="|")
pd.DataFrame(test_data, columns=["Artist", "Title", "Lyrics"]).to_csv('chosen_test.csv', sep="|")

if len(sys.argv) > 1:
    lower = sys.argv[1]
    upper = sys.argv[2]
    TF = sys.argv[3] if len(sys.argv) > 3 else 'regular'

    strain = 'train_' + str(lower) + '-' + str(upper)
    sdev = 'dev_' + str(lower) + '-' + str(upper)
    stest = 'test_' + str(lower) + '-' + str(upper)
    vocab = buildVocabulary(lower, upper)
    featureExtractor(train_data, strain, vocab, lower, upper, TF)
    featureExtractor(dev_data, sdev, vocab, lower, upper, TF, vocab)
    featureExtractor(test_data, stest, vocab, lower, upper, TF, vocab)

else:
    vocab = buildVocabulary(10, 1000)
    featureExtractor(train_data, 'train_data', vocab)
    featureExtractor(train_data, 'dev_data', vocab)
    featureExtractor(test_data, 'test_data', vocab)

