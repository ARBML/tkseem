import os
import pickle
from .__base import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    """
    White space based tokenization 
    """

    tokens_frequency = None

    def train(self, large_file=False):
        """
        Train data using tokens' frequency

        Args:
            large_file (bool, optional): Use memory mapping to read the datta quickly. Defaults to False.
        """
        print("Training WordTokenizer...")
        self._check_train_data_path()
        if large_file:
            sorted_tokens_frequency = {
                k: v
                for k, v in sorted(
                    self._get_tokens_frequency_quickly("data/raw/train.txt").items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            }
        else:
            sorted_tokens_frequency = {
                k: v
                for k, v in sorted(
                    self._get_tokens_frequency("data/raw/train.txt").items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            }

        limited_tokens_frequency = dict()
        limited_tokens_frequency[self.unk_token] = -1
        limited_tokens_frequency[self.pad_token] = -1
        limited_tokens_frequency.update(
            {k: v for k, v in list(sorted_tokens_frequency.items())[: self.vocab_size]}
        )
        self.vocab = limited_tokens_frequency
        self.vocab_size = len(self.vocab)

    def tokenize(self, text):
        """Tokenize using the frequency dictionary 

        Args:
            text (str): input string

        Returns:
            list: generated tokens
        """
        assert self.vocab
        output_tokens = []
        for word in text.split():
            if word in self.vocab.keys():
                output_tokens.append(word)
            else:
                output_tokens.append(self.unk_token)
        return output_tokens

    def detokenize(self, tokens):
        """ Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = " ".join(tokens)
        return detokenized
