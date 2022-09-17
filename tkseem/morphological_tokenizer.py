import os
import pickle

from ._base import BaseTokenizer


class MorphologicalTokenizer(BaseTokenizer):
    """ Auto tokenization using a saved dictionary"""

    def __init__(
        self, vocab_size=10000,
    ):
        super(MorphologicalTokenizer, self).__init__(vocab_size=vocab_size)
        self.name = "MorphologicalTokenizer"
        
    def train(self):
        """Use a default dictionary for training"""
        print("Training MorphologicalTokenizer ...")
        vocab_path = os.path.join(self.rel_path, "dictionaries/vocab.pl")
        self.vocab = self._truncate_dict(pickle.load(open(vocab_path, "rb")))
