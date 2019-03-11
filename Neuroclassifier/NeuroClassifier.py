from typing import List, Dict
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
from clr_callback import CyclicLR
from keras.models import Sequential
from NeuroModels import NeuroModels

np.random.seed(1337)

class NeuroClassifier():
    def __init__(
            self,
            samples: List[str],
            labels: List[int],
            max_vocab_size: int,
            max_seq_len: int,
            model_arch: str,
            validation_split: float=0.2,
            crop_amount: int=None):

        """
        Main class for training the Neuroclassifier
        
        Arguments:
            samples {List[str]} -- List of str for the training data
            labels {List[str]} -- List of int for the labels
            model_arch {str} -- The model architecture
            validation_split {float} -- The percentage of data to use for validation
            crop_amount {int} -- If given, the amount to crop the training data. Useful for quick prototyping.
        """

        self.model_arch = model_arch
        self.validation_split = validation_split
        self.crop_amount = crop_amount

        self.samples = samples
        self.labels = labels

        # Validated data
        self.vocab_size = max_vocab_size
        self.seq_len = max_seq_len

        # Processed data
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.index_to_word = None
        self.tokenizer = None
        self.model = None


    def run(self):
        self._validate_params()
        self._process_data()
        print(self.vocab_size)
        self._get_model()
        # self._train_model()
        if self.model_is_auto:
            self._train_auto_model()
        else:
            self._train_model()


    def _get_model(self):
        model_obj = NeuroModels().get_model(self.model_arch)
        self.model_is_auto = model_obj['auto']
        self.model = model_obj['model']


    def _process_data(self):
        # Declare Keras Tokenizer
        self.tokenizer = Tokenizer(
                    num_words=self.vocab_size, # If self.vocab_size is not None, limit vocab size
                    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'')

        # Build Tokenizer on training vocab
        self.tokenizer.fit_on_texts(self.samples)
        # Tokenize sequences from words to integers
        self.X_train = self.tokenizer.texts_to_sequences(self.samples)
        # Pad sequence to match MAX_SEQ_LEN
        self.X_train = pad_sequences(self.X_train, maxlen=self.seq_len)
        # Index to word
        self.index_to_word = dict(map(reversed, self.tokenizer.word_index.items()))

        # Split data, so it would be shuffeled before cropping
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.labels, test_size=self.validation_split, random_state=42)
        # Crop the training data if crop_amount is given
        if self.crop_amount:
            self.X_train = self.X_train[:self.crop_amount]
            self.y_train = self.y_train[:self.crop_amount]
            self.X_val = self.X_val[:self.crop_amount]
            self.y_val = self.y_val[:self.crop_amount]

        # Change self.vocab_size to the final vocab size, if it was less than the max
        final_vocab_size = len(self.tokenizer.word_index)
        if final_vocab_size < self.vocab_size:
            self.vocab_size = final_vocab_size


    def _train_model(self):
        bs = 64
        # Custom cyclical learning rate callback
        clr_triangular = CyclicLR(mode='triangular', step_size=6*(len(self.X_train)/bs))
        self.model = self.model(self.vocab_size, self.seq_len)
        self.model.fit(self.X_train, self.y_train,
                        batch_size=bs,
                        epochs=5,
                        verbose=2,
                        # validation_split=self.validation_split,
                        validation_data=(self.X_val, self.y_val),
                        callbacks=[clr_triangular]
                        )


    def _train_auto_model(self):
        p = {
            'lr': (0.5, 5, 10),
            'first_neuron': [24, 48, 96],
            'e_size':[128, 300],
            'h_size':[32, 64],
            'hidden_layers':[2,3],
            'activation':[relu],
            'batch_size': [64],
            'epochs': [2, 5],
            'dropout': (0.20, 0.60),
            'optimizer': [Adam, Nadam],
            'seq_len': [self.seq_len],
            'vocab_size': [self.vocab_size],
            'last_activation': [sigmoid]
        }

        h = ta.Scan(self.X_train, self.y_train,
            params=p,
            model=self.model,
            grid_downsample=0.01,
            val_split=0, # Zerofy val_split in Talos, because we pass in our own val data
            x_val=self.X_val,
            y_val=self.y_val
        )

        # Get only the model from the Talos experiment
        best_model = best_model(h, 'val_acc', False)
        self.model = activate_model(h, best_model)

        # For summary
        model_summary = model.summary()
        # For model json object
        model_json = model.to_json()
        print(model_summary)
        print(model_json)

        # For svg of model
        from keras.utils import plot_model
        plot_model(model, to_file='model.png') # TODO proper file path

        # Also TODO https://keras.io/visualization/

    def _data_for_auto_model(self):
        return self.X_train, self.y_train

    def _validate_params(self):
        # TODO validate params
        # If proper, set them
        pass


if __name__ == '__main__':
    CROP = 5000
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
    print(len(pos_y), len(neg_y))
    X = pos_samples + neg_samples
    y = pos_y + neg_y


    ### TEST HERE
    # neuro_classifier = NeuroClassifier(['Hey this is the positive sample', 'Oh and here is the negative one you know'], [1, 0], 100, 3, 'SimpleFNN')
    # neuro_classifier = NeuroClassifier(X, y, 50000, 150, 'simpleFNN', crop_amount=1000)
    # neuro_classifier.run()
    
    neuro_classifier = NeuroClassifier(X, y, 50000, 150, 'autoFNN', crop_amount=1000)
    neuro_classifier.run()
    import pdb;pdb.set_trace()

    # print()
    # print()
    # print()
    # neuro_classifier = NeuroClassifier(X, y, 50000, 150, 'SimpleCNN')
    # neuro_classifier.run()
    # print()
    # print()
    # print()
    # neuro_classifier = NeuroClassifier(X, y, 50000, 150, 'SimpleGRU')
    # neuro_classifier.run()
    # print()
    # print()
    # print()
    # neuro_classifier = NeuroClassifier(X, y, 50000, 150, 'SimpleLSTM')
    # neuro_classifier.run()
    # print()
    # print()
    # print()
    # neuro_classifier = NeuroClassifier(X, y, 50000, 150, 'GRUCNN')
    # neuro_classifier.run()
    # print()
    # print()
    # print()
    # neuro_classifier = NeuroClassifier(X, y, 50000, 150, 'LSTMCNN')
    # neuro_classifier.run()
