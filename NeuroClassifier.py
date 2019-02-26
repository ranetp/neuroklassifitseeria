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
from keras.models import Sequential
from keras.layers import Dense, Embedding, CuDNNLSTM, CuDNNGRU, Dropout


class NeuroClassifier():
    def __init__(
                self,
                samples: List[str],
                labels: List[int],
                max_vocab_size: int,
                max_seq_len: int,
                model_type: str,
                validation_split: float=0.2
                ):
        """
        Main class for training the Neuroclassifier
        
        Arguments:
            samples {List[str]} -- List of str for the training data
            labels {List[str]} -- List of int for the labels
            model_type {str} -- Type of model to be used for the Task
            validation_split {float} -- The percentage of data to use for validation
        """

        self.model_type = model_type
        self.validation_split = validation_split # TODO use in model fit

        self.samples = samples
        self.labels = labels

        # Validated data
        self.vocab_size = max_vocab_size
        self.seq_len = max_seq_len

        # Processed data
        self.X_train = None
        self.index_to_word = None
        self.tokenizer = None

        # TODO
        # Get data / 
        # Tokenize data [params] / 
        # Figure out a system whether to get vocab size automatically or not /
        # Parse params for training methods, like conv1d, grurnn, lstmrnn, dense, etc [params] /
        # Default hyperparams [bool]
        # Automatic hyperparameter search [params]

    def run(self):
        self._validate_params()
        self._process_data()
        model = self._get_model()


    def _get_model(self):
        return Models.get_model(self.model_type, self.vocab_size, self.seq_len)


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

        # Change self.vocab_size to the final vocab size, if it was less than the max
        final_vocab_size = len(self.tokenizer.word_index)
        if final_vocab_size < self.vocab_size:
            self.vocab_size = final_vocab_size
    
    def _validate_params():
        # TODO validate params
        # If proper, set them
        pass


class Models():
    def __init__(self, model_type: str, vocab_size: int, seq_len: int):
        self.models_map = {
            'SimpleFNN': self.SimpleFNN,
            'SimpleCNN': self.SimpleCNN,
            'SimpleGRU': self.SimpleGRU,
            'SimpleLSTM': self.SimpleLSTM,
            'GRUCNN': self.GRUCNN,
            'LSTMCNN': self.LSTMCNN,
            'FullAuto': self.FullAuto,
        }

        self.model_type = model_type
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def get_model(self):
        if self.model_type in self.models_map:
            return self.models_map[self.model_type](self.vocab_size, self.seq_len)
        else:
            raise ValueError(f'{self.model_type} is not a valid model type!')

    # Simplier models
    def SimpleFNN():
        pass

    def SimpleCNN():
        pass

    def SimpleGRU():
        pass
    
    def SimpleLSTM():
        pass
    
    # More complicated
    def GRUCNN():
        pass
    
    def LSTMCNN():
        pass

    # Full search
    def FullAuto():
        pass

