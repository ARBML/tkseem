import os
import pickle
from .__base import BaseTokenizer


class AutoTokenizer(BaseTokenizer):
    """ Auto tokenization using a saved dictionary 
    """

    def train(self):
        """Use a default dictionary for training"""
        print("Training AutoTokenizer...")
        self._check_train_data_path()
        vocab_path = os.path.join(self.rel_path, "dictionaries/vocab.pl")
        self.vocab = self._truncate_dict(pickle.load(open(vocab_path, "rb")))

