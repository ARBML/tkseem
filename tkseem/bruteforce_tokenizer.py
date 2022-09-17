import functools
import operator
import random
from collections import defaultdict

from ._base import BaseTokenizer


class BruteForceTokenizer(BaseTokenizer):
    """ BruteForceTokenizer tokenization 
    """
    def __init__(
        self, vocab_size=10000,
    ):
        super(BruteForceTokenizer, self).__init__(vocab_size=vocab_size)
        self.name = "BruteForceTokenizer"

    def train(self, file_path):
        """Train data using randomly splitted subwords 

        Args:
            file_path (str): file to train 
        """
        
        print("Training RandomTokenizer ...")
        text = open(file_path, "r").read()
        self.vocab = self._truncate_dict(self._bruteforce_dict(text))
        self.vocab_size = len(self.vocab)

    ##TODO too slow we need to speed up
    def _bruteforce_dict(self, text):
        """Create dictionary based on random splitting

        Args:
            text (str): input text

        Returns:
            Dict: tokens frequency
        """

        tokens_frequency = defaultdict(int)
        text = text.replace("\n", "")

        for word in text.split(" "):
            if word.strip() == "":
                continue

            # cached word splitting only accept words with max 20 length
            if len(word) >= 20:
                continue

            # random number of splits
            word = word.strip()
            for i in range(1, len(word)):
                groups = self._split_word_cached(word, i)
                groups = functools.reduce(operator.iconcat, groups, [])

            for sub_word in groups:
                tokens_frequency[sub_word] += 1
        return dict(tokens_frequency)
