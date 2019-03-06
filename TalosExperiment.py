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
from keras.layers import Dense, Embedding, CuDNNLSTM, CuDNNGRU, Dropout, GRU, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam, Nadam
from keras.losses import categorical_crossentropy, logcosh
from keras.activations import relu, elu, softmax, sigmoid

from talos.model.layers import hidden_layers
from talos.model.normalizers import lr_normalizer
from talos import Deploy
from talos.utils.best_model import activate_model, best_model

def data(crop: int):
    ## DATA LOADING
    # Crop data for faster prototyping
    def read_csv(path, crop):
        samples = []
        with open(path, encoding='utf8') as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                if len(samples) < crop:
                    samples.append(row[0])
        return samples
    pos_samples = read_csv('C:/Users/ranet/Documents/DATA/Datasets/farm_not_farm_text_tagging/farmstuff.csv', crop)
    neg_samples = read_csv('C:/Users/ranet/Documents/DATA/Datasets/farm_not_farm_text_tagging/notfarmstuff.csv', crop)
    
    ## MAKE CLASSES
    # Combine both, add classes
    pos_y = [1 for x in range(len(pos_samples))]
    neg_y = [0 for x in range(len(neg_samples))]
    X = pos_samples + neg_samples
    y = pos_y + neg_y

    # X = X[:CROP]
    # y = y[:CROP]

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
    model.add(Conv1D(32, 7, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(32, 7, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(params['dropout']))
    hidden_layers(model, params, 1)
    model.add(Dense(1, activation=params['last_activation']))

    ## COMPILE
    model.compile(optimizer=params['optimizer'](lr_normalizer(params['lr'], params['optimizer'])),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    out = model.fit(X_train, Y_train,
                    batch_size=params['batch_size'],
                    epochs=params['epochs'],
                    validation_data=[X_val, Y_val],
                    verbose=2)

    return out, model


if __name__ == '__main__':

    crop = 300
    X_train, Y_train, vocab_size, seq_len = data(crop)
    print(X_train.shape, Y_train.shape)
    X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)

    # X_train = X_train[:crop]
    # X_val = X_val[:crop]
    # y_train = y_train[:crop]
    # y_val = y_val[:crop]

    p = {
     'lr': (0.5, 5, 10),
     'first_neuron': [24, 48, 96],
     'e_size':[128, 300],
     'gru_h_size':[32, 64],
     'hidden_layers':[2,3],
     'activation':[relu],
     'batch_size': [64, 128],
     'epochs': [2, 5],
     'dropout': (0, 0.20, 0.60),
     'optimizer': [Adam, Nadam],
     'seq_len': [seq_len],
     'vocab_size': [vocab_size],
     'last_activation': [sigmoid]
    }

    h = ta.Scan(X_train, Y_train,
          params=p,
          model=build_model,
          grid_downsample=0.01
        )
    print('done')
    # Might be useful for exporting and easy overview
    # But problem is that its a compressed file right away
    Deploy(h, 'newname')
    # Get only the model
    best_model = best_model(h, 'val_acc', False)
    model = activate_model(h, best_model)

    # For summary
    model.summary()
    # For model object
    model.to_json()

    # For svg of model
    from keras.utils import plot_model
    plot_model(model, to_file='model.png')
    # Also https://keras.io/visualization/
    import pdb;pdb.set_trace()




