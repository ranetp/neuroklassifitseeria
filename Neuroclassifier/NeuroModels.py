from clr_callback import CyclicLR
from keras.models import Sequential
from keras.layers import (Dense,
                          Embedding,
                          CuDNNLSTM,
                          CuDNNGRU,
                          GRU,
                          Bidirectional,
                          Dropout,
                          MaxPooling1D,
                          Conv1D,
                          GlobalAveragePooling1D,
                          MaxPooling1D,
                          Flatten,
                         )

class NeuroModels():
    def __init__(self):
        self.models_map = {
            'SimpleFNN': { 'method': 'manual', 'model': self.SimpleFNN },
            'SimpleCNN': { 'method': 'manual', 'model': self.SimpleCNN },
            'SimpleGRU': { 'method': 'manual', 'model': self.SimpleGRU },
            'SimpleLSTM': { 'method': 'manual', 'model': self.SimpleLSTM },
            'GRUCNN': { 'method': 'manual', 'model': self.GRUCNN },
            'LSTMCNN': { 'method': 'manual', 'model': self.LSTMCNN },
            'AutoFNN': { 'method': 'auto', 'model': self.AutoFNN },
            'FullAuto': { 'method': 'auto', 'model': self.FullAuto },
        }

    def get_model(self, model_type):
        if model_type in self.models_map:
            return self.models_map[model_type]
        else:
            raise ValueError(f'"{model_type}" is not a valid model type!')


    # Simplier models
    @staticmethod
    def SimpleFNN( vocab_size, seq_len):
        embed_dim = 128
        model = Sequential()
        model.add(Embedding(vocab_size, embed_dim, input_length=seq_len))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model


    @staticmethod
    def SimpleCNN():
        embed_dim = 128
        model = Sequential()
        model.add(Embedding(vocab_size, embed_dim, input_length=seq_len))
        model.add(Conv1D(32, 7, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(32, 7, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model


    @staticmethod
    def SimpleGRU():
        embed_dim = 128
        n_hidden = 32
        model = Sequential()
        model.add(Embedding(vocab_size, embed_dim, input_length=seq_len))
        model.add(CuDNNGRU(n_hidden,))
        model.add(Dense(1,activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics = ['accuracy'])
        return model


    @staticmethod
    def SimpleLSTM():
        embed_dim = 128
        n_hidden = 32
        model = Sequential()
        model.add(Embedding(vocab_size, embed_dim, input_length=seq_len))
        model.add(CuDNNLSTM(n_hidden))
        model.add(Dense(1,activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics = ['accuracy'])
        return model


    # Combined models
    @staticmethod
    def GRUCNN():
        embed_dim = 128
        model = Sequential()
        model.add(Embedding(vocab_size, embed_dim, input_length=seq_len))
        model.add(Conv1D(32, 7, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(32, 7, activation='relu'))
        model.add(CuDNNGRU(32))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model


    @staticmethod
    def LSTMCNN():
        embed_dim = 128
        model = Sequential()
        model.add(Embedding(vocab_size, embed_dim, input_length=seq_len))
        model.add(Conv1D(32, 7, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Conv1D(32, 7, activation='relu'))
        model.add(CuDNNLSTM(32))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model

    # Auto models
    @staticmethod
    def AutoFNN(x_train, y_train):
        embed_dim = {{ choice([32, 64, 128, 256, 300]) }}
        model = Sequential()
        model.add(Embedding(vocab_size, embed_dim, input_length=seq_len))
        model.add(Flatten())
        model.add(Dropout( {{ uniform(0, 1) }} ))
        model.add(Dense( {{ choice([32, 64, 128, 256]) }} , activation='relu'))
        model.add(Dropout(  {{ uniform(0, 1) }}  ))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer={{ choice(['rmsprop', 'adam', 'sgd'])}})

        result = model.fit(x_train, y_train,
              batch_size={{ choice([64, 128]) }},
              epochs={{ choice([1, 2, 3]) }},
              verbose=2,
              validation_split=0.1)

        #get the highest validation accuracy of the training epochs
        validation_acc = np.amax(result.history['val_acc']) 
        print('Best validation acc of epoch:', validation_acc)

        return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

    # Full search
    @staticmethod
    def FullAuto():
        pass
