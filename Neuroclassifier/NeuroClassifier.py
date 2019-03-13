from typing import List, Dict

# Data imports
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

# Model imports
from keras.models import Sequential
from keras.activations import relu, elu, softmax, sigmoid
from keras.optimizers import Adam

# Talos imports
import talos as ta
from talos.model.layers import hidden_layers
from talos.model.normalizers import lr_normalizer
from talos import Deploy
from talos.utils.best_model import activate_model, best_model

from clr_callback import CyclicLR
from NeuroModels import NeuroModels

np.random.seed(1337)

class NeuroClassifier():
    def __init__(self, samples: List[str], model_path: str, tokenizer_path: str):
        """
        Classify documents with a trained NeuroClassifier model
        
        Arguments:
            samples {List[str]} -- List of str for the training data
            model {str} -- The path to model and path to metadata such as index_to_word
        """
        self.samples = samples
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        self.model = None
        self.tokenizer = None
        self.index_to_word = None
        self.processed_samples = None


    def run(self):
        self._validate_params()
        self._load_model_and_tokenizer()
        self._process_data()
        self._tag_documents()


    def _load_model_and_tokenizer(self):
        # TODO
        # self.model = pass
        # self.tokenizer = pass

        # Index to word
        self.index_to_word = dict(map(reversed, self.tokenizer.word_index.items()))


    def _process_data(self):
        # TODO
        # Tokenize data
        self.processed_samples = self.tokenizer.texts_to_sequences(self.samples)

    
    def _tag_documents(self):
        # TODO
        preds = self.model.predict(self.processed_samples)


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
    
    X = pos_samples + neg_samples

    neuro_classifier = NeuroClassifier(X, 'modelpath', 'tokpath')
    neuro_classifier.run()
    print('RUN DONE')
    import pdb;pdb.set_trace()
