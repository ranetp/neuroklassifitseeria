# Hyperas import
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

# Data imports
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

# Model imports
from keras.models import Sequential
from keras.layers import Dense, Embedding, CuDNNLSTM, CuDNNGRU, Dropout


def data():
    ## DATA LOADING
    # Crop data for faster prototyping
    CROP = 10000
    def read_csv(path):
        samples = []
        with open(path, encoding='utf8') as csvfile:
            spamreader = csv.reader(csvfile)
            for row in spamreader:
                samples.append(row[0])
        return samples
    pos_samples = read_csv('C:/Users/ranet/Documents/DATA/Datasets/farm_not_farm_text_tagging/farmstuff.csv')[:CROP]
    neg_samples = read_csv('C:/Users/ranet/Documents/DATA/Datasets/farm_not_farm_text_tagging/notfarmstuff.csv')[:CROP]
    
    ## MAKE CLASSES
    # Combine both, add classes
    pos_y = [1 for x in range(len(pos_samples))]
    neg_y = [0 for x in range(len(neg_samples))]
    X = pos_samples + neg_samples
    y = pos_y + neg_y


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
    X_train, X_test, Y_train, Y_test = train_test_split(X_tokenized, y, test_size=0.33, random_state=42)
    
    
    return X_train[:CROP], Y_train[:CROP], X_test[:CROP], Y_test[:CROP], MAX_VOCAB_SIZE, MAX_SEQ_LEN

def model(X_train, Y_train, X_test, Y_test, MAX_VOCAB_SIZE, MAX_SEQ_LEN):
    ## DEFINE MODEL
    embed_dim = 64
    lstm_out = 32

    model = Sequential()
    model.add(Embedding(MAX_VOCAB_SIZE, embed_dim, input_length=MAX_SEQ_LEN))
    if conditional({{ choice(['gru', 'lstm']) }}) == 'gru':
        model.add(CuDNNGRU( {{ choice([8, 32, 64]) }} ))
    else:
        model.add(CuDNNLSTM( {{ choice([8, 32, 64]) }} ))
    
    if conditional({{choice(['one', 'two', 'three'])}}) == 'two':
        model.add(Dense({{choice([5, 20, 50, 100])}}, activation="relu"))
    elif conditional({{choice(['one', 'two', 'three'])}}) == 'three':
        model.add(Dense({{choice([20, 50, 100])}}, activation="relu"))
        model.add(Dropout({{ uniform(0, 1) }}))
        model.add(Dense({{choice([5, 20])}}, activation="relu"))

    model.add(Dense(1, activation='sigmoid'))

    ## COMPILE
    model.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer={{ choice(['rmsprop', 'adam', 'sgd']) }}
                  )

    result = model.fit(X_train, Y_train,
          batch_size={{choice([32, 64, 128])}},
          epochs=5,
          verbose=2,
          validation_data=(X_test, Y_test))

    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test, _, _ = data()
    print("Evalutation of best performing model:")
    print(best_model.metrics_names)
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print("Best model summary:")
    print(best_model.summary())


