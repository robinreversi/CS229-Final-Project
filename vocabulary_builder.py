
# Vocabulary Builder for CS 229 Final Project: Whose Rap is it Anyways?
# Alex Wang, Robin Cheong, Vince Ranganathan
# jwang98, robinc20, akranga @ stanford.edu
# Updated 11/04/2017
from nltk.stem import PorterStemmer
import pandas as pd
import unicodedata
import regex as re
ps = PorterStemmer()

#----------------------------------#

def buildVocabulary(lower, upper, filename):

    def preprocessText(input):
        '''
        :param input: verse as a single string, (delimited by '\n'?)
        :return: list where elements are lines of the verse
        '''
        words = input.decode('utf-8').split()
        return [ps.stem(word) for word in words]

    data = pd.read_csv(filename, delimiter='|')
    lyrics = data['Lyrics'].values

    # With frequency filtering

    vocab = {}

    for song in lyrics:
        text = preprocessText(song)
        for word in text:
            word = re.sub(ur"\p{P}+", "", word)
            vocab[word] = vocab.get(word, 0) + 1

    dic = {k:v for k, v in vocab.items() if (int(lower) <= v <= int(upper)) }

    return set(dic.keys())

#-----------------------------------#

#print(list(buildVocabulary(3, 1000)))
