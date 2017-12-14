
# Vocabulary Builder for CS 229 Final Project: Whose Rap is it Anyways?
# Alex Wang, Robin Cheong, Vince Ranganathan
# jwang98, robinc20, akranga @ stanford.edu
# Updated 11/04/2017
from nltk.stem import PorterStemmer
import pandas as pd
import unicodedata
import seaborn as sns
import regex as re
import matplotlib.pyplot as plt


sns.set()
sns.set_style("white")
sns.set_context("talk")
#sns.set_palette(sns.color_palette([(254, 255, 149), 'red']))
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
            word = re.sub(r'\p{P}+', "", word)
            vocab[word] = vocab.get(word, 0) + 1


    '''        
    plt.hist(vocab.values(), bins=range(50, 1550, 100), color=[(254.0/255, 1, 149.0/255)])

    plt.ylabel('Number of Words')
    plt.xlabel('Frequency')
    plt.axvline(x=10, color='red')
    plt.title('Number of Words vs. Frequency')

    plt.show()
    '''
    dic = {k:v for k, v in vocab.items() if (int(lower) <= v <= int(upper)) }
    
    '''
    plt.hist(dic.values(), bins=range(50, 2000, 50), color=[(254.0/255, 1, 149.0/255)])
    plt.ylabel('Number of Words')
    plt.xlabel('Frequency')
    plt.axvline(x=1000, color='red')
    plt.title('Number of Words vs. Frequency')

    plt.show()
    '''
    return set(dic.keys())

#-----------------------------------#

#print(len(buildVocabulary(10, 1000, 'chosen_train.csv')))
