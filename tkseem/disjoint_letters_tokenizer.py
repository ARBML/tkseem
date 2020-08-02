import re
import pickle
from collections import defaultdict
from .__base import BaseTokenizer


class DisjointLetterTokenizer(BaseTokenizer):
    """ Disjoint Letters based tokenization 
    """

    def train(self):
        """Train data using disjoint letters
        """
        print("Training DisjointLetterTokenizer ...")
        self._check_train_data_path()
        rx = re.compile(r"([اأإآءؤﻵﻹﻷدذرزو])")

        text = open("data/raw/train.txt", "r").read()
        text = rx.sub(r"\1## ", text)
        text = text.replace("## ", " ##")

        tokens_frequency = defaultdict(int)
        for word in text.split(" "):
            tokens_frequency[word] += 1

        self.vocab = self._truncate_dict(dict(tokens_frequency))
        self.vocab_size = len(self.vocab)

