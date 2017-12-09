
# Feature extractor for CS 229 Final Project: Whose Rap is it Anyways?
# Alex Wang, Robin Cheong, Vince Ranganathan
# jwang98, robinc20, akranga @ stanford.edu
# Updated 11/02/2017

from nltk.stem import PorterStemmer as ps
from vocabulary_builder import buildVocabulary
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

ps = ps()


#----------------------------------#

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

def featureExtractor(raw_data, filename, lower=0, upper=20000, verbose=0):
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
    vocab = buildVocabulary(lower, upper)
    processed_data = []

    # data_pt[0] = artist
    # data_pt[1] = song name
    # data_pt[2] = lyrics
    #print len(raw_data)
    for data_pt in raw_data:
        #print data_pt
        artist = artist_map[data_pt[0]]
        vocab_dict = dict.fromkeys(vocab, 0)
        #print data_pt
        for word in preprocessText(data_pt[2]):
            if(word in vocab_dict):
                vocab_dict[word] += 1
        phi = ([1] + list(vocab_dict.values()))
        processed_data.append([artist] + phi)
    processed_df = pd.DataFrame(processed_data)
    #print processed_df.head()
    processed_df.to_csv(filename)


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
data = pd.read_csv('data_scraping/songs.csv', delimiter='|')
raw_data, test_data = train_test_split(data,test_size=0.33, random_state=420)
raw_data = raw_data.as_matrix()
test_data = test_data.as_matrix()

#print test_data
if len(sys.argv) > 1:
    lower = sys.argv[1]
    upper = sys.argv[2]

    strain = 'train_data_' + str(lower) + '-' + str(upper) + '.csv'
    stest = 'test_data_' + str(lower) + '-' + str(upper) + '.csv'

    featureExtractor(raw_data, strain, lower, upper)
    featureExtractor(test_data, stest, lower, upper)

else:
    featureExtractor(raw_data, 'train_data_freqfilter.csv')
    featureExtractor(test_data, 'test_data_freqfilter.csv')

