
# Feature extractor for CS 229 Final Project: Whose Rap is it Anyways?
# Alex Wang, Robin Cheong, Vince Ranganathan
# jwang98, robinc20, akranga @ stanford.edu
# Updated 11/01/2017

def featureExtractor(input):
    '''
    ---FEATURES 'KEYNAME' : DESCRIPTION---
    -Intercept 'incpt': 1

    --Vocabulary
    -Vocabulary density 'vocabdense': (float) fraction of words that are unique
    -Vocabulary 'vocab': (dict) map of (number of words) : (number of occurrences of word)
    -Abusive language distribution 'abusivelangdist': (list) vector representing distribution of number of abusive words per sentence

    --Length
    -Line length distribution 'linelendist': (list) vector representing distribution of line length in words
    -Word length distribution 'wordlendist': (list) vector representing distribution of word length in chars
    -Verse length 'verselength': (int) length of the verse in lines

    --Rhyme
    -Weak rhyme approximation 'weakrhymes': (int) number of weak rhymes (up to 2 lines ahead, exactly last two chars match)
    -Strong rhyme approximation 'strongrhymes': (int) number of strong rhymes (up to 2 lines ahead, exactly last three chars match)

    :param input: a string of text (with lines delimited by '\n'?) representing a single verse
    :return: the feature vector for the input
    '''

    abusiveLang = {}  # TODO put in a list of bad words?

    def convertToList(input):
        '''
        :param input: verse in some text format (delimited by '\n'?)
        :return: list where elements are lines of the verse
        '''
        return input.splitlines()

    def initialize(phi, verse):
        '''
        :param phi: as given
        :param verse: as given
        :return: n/a (phi initialized)
        '''
        # initializing the feature vector as a dict
        phi.update({'incpt': 1, 'verselength': len(verse)})

        # setting up the distributions as dicts
        phi.update({'vocab': {}})
        phi.update({'linelendist': {}})
        phi.update({'wordlendist': {}})
        phi.update({'abusivelangdist': {}})

    def vocabularyAndLength(phi, verse):
        '''
        :param phi: as given
        :param verse: as given
        :return: n/a (Vocabulary and Length features of phi updated)
        '''
        totalWords = 0
        for str in verse:
            # convert to list of words
            line = str.lower().split()  # TODO remove punctuation

            # add line length to distribution
            phi['linelendist'][len(line)] = phi['linelendist'].get(len(line), 0) + 1

            # add amount of abusive language to distribution
            abuses = len([w for w in line if w in abusiveLang])
            phi['abusivelangdist'][abuses] = phi['abusivelangdist'].get(abuses, 0) + 1

            for word in line:
                # add word to vocab
                phi['vocab'][word] = phi['vocab'].get(word, 0) + 1

                # add word length to distribution
                phi['wordlendist'][len(word)] = phi['wordlendist'].get(len(word), 0) + 1

            # update word count
            totalWords += len(line)

        # add fraction of words that are unique = number of unique words / total number of words
        phi.update({'vocabdense': len(phi['vocab'].keys()) / float(totalWords)})

    def rhyme(phi, verse, lookahead=2):
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
                        phi['strongrhymes'] = phi.get('strongrhymes', 0) + 1
                        break
                    else:
                        phi['weakrhymes'] = phi.get('weakrhymes', 0) + 1
                        break
        return

    #######################################

    # setup
    verse = convertToList(input)
    phi = {}
    initialize(phi, verse)

    # feature extraction
    vocabularyAndLength(phi, verse)
    rhyme(phi, verse)

    return phi

"""
input = "The world is spinning\nThe days are changing\nThe lives are hungry\nAnd I am so funny\nI like to head food\nDoes that rhyme with rod\nHello all my friends"
phi = featureExtractor(input)
print(phi)
"""
