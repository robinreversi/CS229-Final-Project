from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import TimeDistributed
import numpy as np
import pandas as pd

# define documents
train_data = pd.read_csv('chosen_train.csv',sep='|')
dev_data = pd.read_csv('chosen_dev.csv',sep='|')
test_data = pd.read_csv('chosen_test.csv',sep='|')
# define class labels
def getArtists():
    with open('artists.txt') as f:
        return f.read().splitlines()
artist_list = getArtists()
def art_id(art):
    return artist_list.index(art)

def onehot(y):
    ret = np.zeros((len(y),12))
    for i in range(len(y)):
        num = y[i]
        ret[i,int(num)] = 1.0
    return ret

train_y = onehot(train_data['Artist'].apply(art_id))
dev_y = onehot(dev_data['Artist'].apply(art_id))
test_y = onehot(test_data['Artist'].apply(art_id))


train_x = train_data['Lyrics'].values
dev_x = train_data['Lyrics'].values
test_x = train_data['Lyrics'].values


# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(train_x)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs_train = t.texts_to_sequences(train_x)
encoded_docs_dev = t.texts_to_sequences(dev_x)
encoded_docs_test = t.texts_to_sequences(test_x)
# pad documents to a max length of 4 words
max_length = 5000
padded_trainx = pad_sequences(encoded_docs_train, maxlen=max_length)
padded_devx = pad_sequences(encoded_docs_dev, maxlen=max_length)
padded_testx = pad_sequences(encoded_docs_test, maxlen=max_length)
print(padded_trainx.shape)
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt',encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length,trainable=False,weights=[embedding_matrix]))
model.add(LSTM(200,return_sequences=False))
model.add(Dense(36))
model.add(Dense(12))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(padded_trainx, train_y, epochs=10, batch_size=50)
# Final evaluation of the model
scores = model.evaluate(padded_devx, dev_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))