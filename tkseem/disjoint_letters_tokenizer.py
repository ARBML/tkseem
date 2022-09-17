import pickle
import re
from collections import defaultdict

from ._base import BaseTokenizer


class DisjointLetterTokenizer(BaseTokenizer):
    """ Disjoint Letters based tokenization 
    """
    def __init__(
        self, vocab_size=10000,
    ):
        super(DisjointLetterTokenizer, self).__init__(vocab_size=vocab_size)
        self.name = "DisjointLetterTokenizer"

    def train(self, file_path):
        """Train data using disjoint letters

        Args:
            file_path (str): file to train
        """
        print("Training DisjointLetterTokenizer ...")
        rx = re.compile(r"([اأإآءؤﻵﻹﻷدذرزو])")

        text = open(file_path, "r").read()
        text = rx.sub(r"\1## ", text)
        text = text.replace("## ", " ##")

        tokens_frequency = defaultdict(int)
        for word in text.split(" "):
            tokens_frequency[word] += 1

        self.vocab = self._truncate_dict(dict(tokens_frequency))
        self.vocab_size = len(self.vocab)
