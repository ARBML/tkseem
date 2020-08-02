import re
import pickle
from collections import defaultdict
from .__base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    """ Character based tokenization 
    """

    def train(self):
        """Train data using characters 
        """
        print("Training CharacterTokenizer ...")
        self._check_train_data_path()
        rx = re.compile(r"\B(.)")

        text = open("data/raw/train.txt", "r").read()
        text = rx.sub(r" ##\1", text)

        tokens_frequency = defaultdict(int)
        for word in text.split(" "):
            tokens_frequency[word] += 1

        self.vocab = self._truncate_dict(dict(tokens_frequency))
        self.vocab_size = len(self.vocab)

    def tokenize(self, text):
        """Tokenize using the frequency dictionary 

        Args:
            text (str): input string

        Returns:
            list: generated tokens
        """
        rx = re.compile(r"\B(.)")
        text = rx.sub(r" ##\1", text)
        output_tokens = []

        for token in text.split():
            if token in self.vocab:
                output_tokens.append(token)
            else:
                output_tokens.append(self.pad_token)
        return output_tokens

