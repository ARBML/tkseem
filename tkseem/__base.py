import mmap
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from farasa.segmenter import FarasaSegmenter
from tqdm import tqdm

from .util import split_on_binary


class BaseTokenizer:
    """
    Base Tokenizer that implements the basic functionalities of a tokenizer
    """

    def __init__(
        self, unk_token="<UNK>", pad_token="<PAD>", vocab_size=10000, special_tokens=[],
    ):
        """Constructor

        Args:
            unk_token (str, optional): reserved token for unknowns. Defaults to "<UNK>".
            pad_token (str, optional): reserved token for padding. Defaults to "<PAD>".
            max_tokens (int, optional): max number of vocabulary. Defaults to 10000.
        """
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.special_tokens = special_tokens

        self.rel_path = os.path.dirname(__file__)
        cach_dict_path = os.path.join(self.rel_path, "dictionaries/cached.pl")
        self.cached = pickle.load(open(cach_dict_path, "rb"))

    def _get_tokens_frequency(self, file_path):
        """
        Get tokens frequency using a dictionary

        Args:
            file_path (str): file path to read
        Returns:
            dict : dict containing frequency
        """
        text = open(file_path, "r").read()
        tokens_frequency = defaultdict(int)
        for word in text.split(" "):
            tokens_frequency[word] += 1
        return dict(tokens_frequency)

    def _get_tokens_frequency_quickly(self, file_path):
        """
        Get the tokens frequency quickly using memory mapping

        Args:
            file_path (str): the directory of the data to read
        
        Returns:
            Dict: frequency based dictionary   
        """
        encoding = "utf8"
        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                m.read(0)
                i = 0
                size_to_read = int(1e9)
                freq = Counter([])
                pbar = tqdm(total=int(m.size() / size_to_read))
                while i < m.size():
                    cur_txt = ""
                    data = m.read(size_to_read)
                    i += size_to_read
                    try:
                        cur_txt = data.decode(encoding)
                    except:
                        cur_txt = (data + m.read(1)).decode(encoding)
                        i += 1
                    freq.update(cur_txt.split(" "))
                    pbar.update(1)
        return freq

    def _write_data(self, path, data):
        """
        Write the string data to a path

        Args:
            file_path (str): the directory of the data to read
        
        """
        return open(path, "w").write(data)

    # I think these two functions should be in the util.
    def _split_word(self, word, number_of_subwords):
        """Split a word into a specific number of sub-words

        Args:
            word (str): word input
            number_of_subwords (int): number of subtokens to generate from the word 
        
        Returns:
            list: list of subwords 
        """
        assert number_of_subwords > 0

        def _split(_word, _number_of_subwords):
            groups = []
            if _number_of_subwords == 1:
                groups.append(["##" + _word])
            else:
                for i in range(1, len(_word), 1):
                    groups.extend(
                        ["##" + _word[:i], *group]
                        for group in _split(_word[i:], _number_of_subwords - 1)
                        if len(group) == _number_of_subwords - 1
                    )
            return groups

        groups_of_subwords = _split(word, number_of_subwords)
        out_groups = []
        for group in groups_of_subwords:
            group[0] = group[0].replace("##", "")
            out_groups.append(group)
        return out_groups

    def _split_word_cached(self, word, number_of_subwords):
        """Faster version of word splitting

        Args:
            word (word): word to be split
            number_of_subwords (int): number of subwords to split the word to

        Returns:
            list: subwords
        """
        if number_of_subwords == 1:
            return [[word]]
        n = len(word) - 1
        all_binaries = self.cached[n, number_of_subwords - 1]
        return [split_on_binary(word, binary) for binary in all_binaries]

    def _tokenize_from_dict(self, text, freq_dict, cache=False):
        """Tokenize using the frequency dictionary 

        Args:
            text (str): input string

        Returns:
            list: generated tokens
        """
        assert freq_dict
        tokens = []
        output_tokens = []
        for word in text.split():
            if word in freq_dict:
                output_tokens.append(word)
            else:
                for i in range(2, len(word) + 1, 1):
                    if cache:
                        groups_of_subwords = self._split_word_cached(word, i)
                    else:
                        groups_of_subwords = self._split_word(word, i)

                    # filter out groups
                    groups_of_valid_subwords = list(
                        filter(
                            lambda group: all(
                                subword in freq_dict.keys() for subword in group
                            ),
                            groups_of_subwords,
                        )
                    )
                    if groups_of_valid_subwords:
                        break
                if len(groups_of_valid_subwords) == 0:
                    output_tokens.append(self.unk_token)
                else:
                    sorted_groups_of_valid_subwords = sorted(
                        groups_of_valid_subwords,
                        key=lambda group: sum(freq_dict[subword] for subword in group),
                    )
                    tokens = sorted_groups_of_valid_subwords[-1]
                    for token in tokens:
                        output_tokens.append(str(token))
        return output_tokens

    def _truncate_dict(self, freq_dict):
        """Truncate a frequency dictionary and add reserved tokens

        Args:
            freq_dict (dict): frequency dictionary

        Returns:
            dict: truncated dictionary based on the vocab size
        """
        sorted_tokens_frequency = {
            k: v for k, v in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        }

        limited_tokens_frequency = dict()
        limited_tokens_frequency[self.unk_token] = -1
        limited_tokens_frequency[self.pad_token] = -1
        limited_tokens_frequency.update(
            {k: v for k, v in list(sorted_tokens_frequency.items())[: self.vocab_size]}
        )
        return limited_tokens_frequency

    def load_model(self, file_path):
        """Load a saved model as a frequency dictionary

        Args:
            file_path (str): file path of the dictionary
        """
        print("Loading as pickle file ...")
        self.vocab = pickle.load(open(file_path, "rb"))

    def save_model(self, file_path):
        """Save a model as a freqency dictionary

        Args:
            file_path (str): file path to save the model
        """
        assert self.vocab
        with open(f"{file_path}", "wb") as pickle_file:
            print("Saving as pickle file ...")
            pickle.dump(self.vocab, pickle_file)

    def tokenize(self, text, cache=False):
        """Tokenize using the frequency dictionary 

        Args:
            text (str): input string

        Returns:
            list: generated tokens
        """
        output_tokens = self._tokenize_from_dict(text, self.vocab, cache)
        return output_tokens

    def detokenize(self, tokens):
        """ Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        # this is the common behaviour along all tokenizers
        detokenized = "".join(tokens).replace("##", "")
        return detokenized

    @property
    def tokens_list(self):
        """
        Get tokens list

        Returns:
            list: tokens 
        """
        if not self.vocab:
            raise NotImplementedError("Please train first to build the vocab list")
        return list(self.vocab.keys())

    def token_to_id(self, piece):
        """ Get tokens list

        Returns:
            list: tokens 
        """
        return list(self.vocab.keys()).index(piece)

    def id_to_token(self, id):
        """ Get tokens list

        Returns:
            list: tokens 
        """
        return list(self.vocab.keys())[id]

    def encode(self, text):
        """ Convert string to a list of ids

        Args:
            text (str): input string

        Returns:
            list: list of ids
        """
        tokens = self.tokenize(text)
        encoded = [self.tokens_list.index(token) for token in tokens]
        return encoded

    def decode(self, encoded):
        """ Decode ids

        Args:
            encoded (list): list of ids to decode

        Returns:
            list: tokens
        """
        decoded = [self.tokens_list[id] for id in encoded]
        return decoded

    def encode_and_save(self, output_directory="data/encoded"):
        """
        Encode all the files then save as numpy.

        Args:
            output_directory (str): frequency dictionary. The output directory will be created if it is not there.

        Returns:
            str: output directory path
        """
        output_directory = output_directory.strip("/")
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        for file_path in os.listdir(output_directory):
            ids = self.encode(open(f"{output_directory}/{file_path}", "r").read())
            np.save(f"{output_directory}/{file_path[:-4]}.npy", ids)
        return output_directory

    def encode_sentences(self, sentences, max_length=20):
        """
        Encode a list of sentences using the trained model

        Args:
            sentences (list): list of sentences
            max_length (int, optional): specify the max length of encodings. Defaults to 100.

        Returns:
            [np.array]: numpy array of encodings
        """
        encodings = []
        for sent in sentences:
            tokens = self.tokenize(sent)
            encoded = []
            for i in range(max_length):
                if i < len(tokens):
                    current_token = tokens[i]
                else:
                    current_token = self.pad_token
                encoded.append(self.tokens_list.index(current_token))
            encodings.append(encoded)
        return np.array(encodings)

    def __str__(self):
        return f"{self.__class__.__name__} tokenizer"
