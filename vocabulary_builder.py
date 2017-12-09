
# Vocabulary Builder for CS 229 Final Project: Whose Rap is it Anyways?
# Alex Wang, Robin Cheong, Vince Ranganathan
# jwang98, robinc20, akranga @ stanford.edu
# Updated 11/04/2017

import pandas as pd
import unicodedata

from nltk.stem import PorterStemmer
import operator
ps = PorterStemmer()

#----------------------------------#

def buildVocabulary(lower, upper):

    def preprocessText(input):
        '''
        :param input: verse as a single string, (delimited by '\n'?)
        :return: list where elements are lines of the verse
        '''
        words = input.decode('utf-8').split()
        return [ps.stem(word) for word in words]

    data = pd.read_csv("data_scraping/songs.csv", delimiter='|')
    lyrics = data['Lyrics'].values

    '''
    vocab = set()

    for song in lyrics:
        # unicodedata.normalize('NFKD', song).encode('ascii', 'ignore')
        text = preprocessText(song)
        for word in text:
            vocab.add(word)

    print(len(vocab))

    return vocab
    '''

    # With frequency filtering

    vocab = {}

    for song in lyrics:
        text = preprocessText(song)
        for word in text:
            vocab[word] = vocab.get(word, 0) + 1

    dic = { k:v for k, v in vocab.items() if (int(lower) <= v <= int(upper)) }

    return set(dic.keys())


    #return set(dict(sorted(vocab.iteritems(), key=operator.itemgetter(1), reverse=True)[k:]).keys())




#-----------------------------------#


#print(buildVocabulary())
