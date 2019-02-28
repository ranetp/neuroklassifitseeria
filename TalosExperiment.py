# Import talos
import talos as ta

# Data imports
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

# Model imports
from keras.models import Sequential
from keras.layers import Dense, Embedding, CuDNNLSTM, CuDNNGRU, Dropout, GRU
from keras.optimizers import Adam, Nadam
from keras.losses import categorical_crossentropy, logcosh
from keras.activations import relu, elu, softmax, sigmoid

def data(CROP: int):
    ## DATA LOADING
    # Crop data for faster prototyping
    def read_csv(path):
        samples = []
        with open(path, encoding='utf8') as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                samples.append(row[0])
        return samples
    pos_samples = read_csv('C:/Users/ranet/Documents/DATA/Datasets/farm_not_farm_text_tagging/farmstuff.csv')
    neg_samples = read_csv('C:/Users/ranet/Documents/DATA/Datasets/farm_not_farm_text_tagging/notfarmstuff.csv')
    
    ## MAKE CLASSES
    # Combine both, add classes
    pos_y = [1 for x in range(len(pos_samples))]
    neg_y = [0 for x in range(len(neg_samples))]
    X = pos_samples + neg_samples
    y = pos_y + neg_y

    X = X[:CROP]
    y = y[:CROP]

    ## TOKENIZER
    MAX_VOCAB_SIZE = 50000
    MAX_SEQ_LEN = 150

    # Declare Keras Tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE,
                           lower=True,
                           split=" ",
                           char_level=False,
                           oov_token="X")

    # Build Tokenizer on training vocab
    tokenizer.fit_on_texts(X)
    # Tokenize sequences from words to integers
    X_tokenized = tokenizer.texts_to_sequences(X)
    # Pad sequence to match MAX_SEQ_LEN
    X_tokenized = pad_sequences(X_tokenized, maxlen=MAX_SEQ_LEN)
    ## SPLIT DATA

    return X_tokenized, np.expand_dims(np.asarray(y), axis=-1), MAX_VOCAB_SIZE, MAX_SEQ_LEN


def build_model(X_train, Y_train, X_val, Y_val, params):
    model = Sequential()
    model.add(Embedding(params['vocab_size'], params['e_size'], input_length=params['seq_len']))
    model.add(CuDNNGRU(params['gru_h_size']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(1, activation=params['last_activation']))

    ## COMPILE
    model.compile(optimizer=params['optimizer'](),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    out = model.fit(X_train, Y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    validation_data=[X_val, Y_val],
                    verbose=0)

    return out, model


if __name__ == '__main__':

    X_train, Y_train, vocab_size, seq_len = data(10000)
    print(X_train.shape)
    print(Y_train.shape)
    print(Y_train[0])

    p = {'lr': (2, 10, 30),
     'e_size':[64, 128, 300],
     'gru_h_size':[4, 8, 16, 32, 64, 128],
     'hidden_layers':[2,3,4,5,6],
     'batch_size': [32, 64, 128, 256],
     'epochs': [2, 5, 7],
     'dropout': (0, 0.20, 0.40, 10),
     'optimizer': [Adam, Nadam],
     'seq_len': [seq_len],
     'vocab_size': [vocab_size],
     'last_activation': [sigmoid]
    }

    h = ta.Scan(X_train, Y_train,
          params=p,
          model=build_model,
          grid_downsample=0.5,
          val_split=0.3)




