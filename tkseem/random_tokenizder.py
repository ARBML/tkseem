import pickle
import random
import operator
import functools
from .__base import BaseTokenizer
from collections import defaultdict


class RandomTokenizer(BaseTokenizer):
    """ Randomized based tokenization 
    """

    def train(self):
        """Train data using randomly splitted subwords 
        """
        print("Training RandomTokenizer ...")
        self._check_train_data_path()
        text = open("data/raw/train.txt", "r").read()
        self.vocab = self._truncate_dict(self._random_dict(text))
        self.vocab_size = len(self.vocab)

    ##TODO too slow we need to speed up
    def _random_dict(self, text):
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
            groups = self._split_word_cached(word.strip(), random.randint(1, len(word)))
            groups = functools.reduce(operator.iconcat, groups, [])

            for sub_word in groups:
                tokens_frequency[sub_word] += 1
        return dict(tokens_frequency)
