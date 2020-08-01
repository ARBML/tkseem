import io
import re
import os
import sys
import mmap
import pickle
import random
import operator
import functools
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sentencepiece as spm
from collections import defaultdict, Counter
from farasa.segmenter import FarasaSegmenter
from .util import clean_data, normalize_data, split_on_binary
from .__base import BaseTokenizer


class AutoTokenizer(BaseTokenizer):
    """ Auto tokenization using a saved dictionary 
    """

    def train(self):
        """Use a default dictionary for training"""
        print("Training AutoTokenizer...")
        vocab_path = os.path.join(self.rel_path, "dictionaries/vocab.pl")
        self.vocab = self._truncate_dict(pickle.load(open(vocab_path, "rb")))

    def tokenize(self, text, cache=False):
        """Tokenize using the frequency dictionary 

        Args:
            text (str): input string

        Returns:
            list: generated tokens
        """
        output_tokens = self._tokenize_from_dict(text, self.vocab, cache)
        return output_tokens

    def _tokens_list(self):
        """ Get tokens list

        Returns:
            list: list of tokens.
        """
        return list(self.vocab.keys())

    def decode(self, encoded):
        """ Decode ids

        Args:
            encoded (list): list of ids to decode

        Returns:
            list: tokens
        """
        decoded = [self._tokens_list()[id] for id in encoded]
        return decoded

    def encode(self, text):
        """ Convert string to a list of ids

        Args:
            text (str): input string

        Returns:
            list: list of ids
        """
        tokens = self.tokenize(text)
        encoded = [self._tokens_list().index(token) for token in tokens]
        return encoded

    def detokenize(self, tokens):
        """ Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = "".join(tokens).replace("##", "")
        return detokenized
