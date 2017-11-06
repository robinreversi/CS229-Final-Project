
# Feature extractor for CS 229 Final Project: Whose Rap is it Anyways?
# Alex Wang, Robin Cheong, Vince Ranganathan
# jwang98, robinc20, akranga @ stanford.edu
# Updated 11/02/2017

from nltk.stem import PorterStemmer
ps = PorterStemmer()

#----------------------------------#

def featureExtractor(input, verbose=0):
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

    def preprocessText(input):
        '''
        :param input: verse as a single string, (delimited by '\n'?)
        :return: list where elements are lines of the verse
        '''
        def stemInput(input):
            words = input.split()
            return [ps.stem(word) for word in words]

        lines = input.splitlines()
        return [" ".join(stemInput(line.lower())) for line in lines]

    def initialize(phi, verse):
        '''
        :param phi: as given
        :param verse: as given
        :return: n/a (phi initialized)
        '''
        # initializing the feature vector as a dict
        phi['_incpt'] = 1
        phi['_verseLen'] = len(verse)

    def Vocabulary_Length(phi, verse, nFeats=8):
        '''
        :param phi: as given
        :param verse: as given
        :return: n/a (Vocabulary and Length features of phi updated)
        '''
        totalWords = 0
        for str in verse:
            # convert to list of words
            line = str.split()  # TODO remove punctuation if desired

            # stems list of words before use
            line = [word for word in line]

            # add line length to distribution
            phi['_linesLen%d' % len(line)] = phi.get('_linesLen%d' % len(line), 0) + 1

            # add amount of abusive language to distribution
            abuses = len([w for w in line if w in abusiveLang])
            phi['_nAbuses%d' % abuses] = phi.get('_nAbuses%d' % abuses, 0) + 1

            for word in line:
                # add word to vocab
                phi['%s' % word] = phi.get('%s' % word, 0) + 1

                # add word length to distribution
                phi['_wordsLen%d' % len(word)] = phi.get('_wordsLen%d' % len(word), 0) + 1

            # update word count
            totalWords += len(line)

        # add fraction of words that are unique = number of unique words / total number of words
        phi['_vocabRich'] = (len(phi.keys()) - nFeats) / float(totalWords)

    def Rhyme(phi, verse, lookahead=2):
        '''
        :param phi: as given
        :param verse: as given
        :param lookahead: number of lines ahead to consider for a rhyme (default is two lines)
        :return: n/a (Rhyme features of phi updated)
        '''
        # iterating through all lines with at least 1 lookahead
        for lineNo in range(len(verse) - 1):
            linesLeft = len(verse) - lineNo - 1

            # iterating through all the NEXT remaining lines within the lookahead
            for compLineNo in range(lineNo + 1, lineNo + min(lookahead, linesLeft) + 1):
                # matching last two chars
                if verse[lineNo][-2:] == verse[compLineNo][-2:]:
                    # third-last char as well?
                    if verse[lineNo][-3] == verse[compLineNo][-3]:
                        phi['_nStrongRhymes'] = phi.get('_nStrongRhymes', 0) + 1
                        break
                    else:
                        phi['_nWeakRhymes'] = phi.get('_nWeakRhymes', 0) + 1
                        break
        return

    #--------EXECUTE--------#

    # setup
    verse = preprocessText(input)
    if verbose: print(verse)

    phi = {}
    initialize(phi, verse)

    # feature extraction
    Vocabulary_Length(phi, verse)
    Rhyme(phi, verse)

    return phi

#----------------------------------#

def test():
    input = "The world is spinning\n" \
            "The days are changing\n" \
            "The lives are hungry\n" \
            "And I am so funny\n" \
            "I like to eat food\n" \
            "Does that rhyme with rod\n" \
            "Fucking hell dude\n" \
            "Hello all my friends"
    phi = featureExtractor(input, verbose=1)
    print(phi)

test()
