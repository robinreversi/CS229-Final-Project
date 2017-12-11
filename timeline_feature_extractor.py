import numpy as np
import pandas as pd


def featureExtractor(raw_data, filename, vocab, lower=0, upper=20000, TF='regular', verbose=0):
    processed_data = []
    K = 0.5

    for data_pt in raw_data:
        print data_pt
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
        processed_data.append([artist] + phi)

    if(TF != 'binary'):
        processed_data = normalize(np.array(processed_data))

    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(filename + '_' + TF + '.csv')
