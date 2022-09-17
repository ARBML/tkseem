import os
import sys
import mmap
import pickle
import itertools
import numpy as np
from tqdm.notebook import tqdm
from pathlib import Path
from .util import split_on_binary
from collections import Counter, defaultdict
from .const import *

class BaseTokenizer:
    """
    Base Tokenizer that implements the basic functionalities of a tokenizer
    """

    def __init__(
        self, vocab_size=10000,
    ):
        """Constructor

        Args:
            vocab_size (int, optional): max vocab size. Defaults to 10000.
        """
        self.vocab_size = vocab_size
        self.pad_idx = 0
        self.unk_idx = 1
        self.sow_idx = 2
        self.sos_idx = 3
        self.eos_idx = 4
        self.special_tokens = [PAD, UNK, SOW, SOS, EOS]
        self.vocab = [PAD, UNK, SOW, SOS, EOS]
        self.sow = SOW
        self.sos = SOS
        self.eos = EOS
        self.pad = PAD
        self.unk = UNK
        self.rel_path = os.path.dirname(__file__)
        cach_dict_path = os.path.join(self.rel_path, "dictionaries/cached.pl")
        self.cached = pickle.load(open(cach_dict_path, "rb"))

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
        words = text.split(" ")
        pbar = tqdm(total=len(words)) 
        for word in words:
            tokens_frequency[word] += 1
            pbar.update(1)
        pbar.close()
        return dict(tokens_frequency)

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

    def _tokenize_from_dict_deprecated(self, text, freq_dict, cache=False, max_size=20):
        """Tokenize using frequency based approach given a dictionary

        Args:
            text (str): input string
            freq_dict (dict): frequency dictionary
            cache (bool, optional): faster approach. Defaults to False.
            max_size (int, optional): maximum word size. Defaults to 20.

        Returns:
            [type]: [description]
        """
        assert freq_dict
        tokens = []
        output_tokens = []
        for word in text.split():
            if len(word) >= max_size:
                print(f"{word} is too long ...")
                output_tokens.append(self.unk)
                continue
            if word in freq_dict:
                output_tokens.append(word)
            else:
                groups_of_valid_subwords = []
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
                    output_tokens.append(self.unk)
                else:
                    sorted_groups_of_valid_subwords = sorted(
                        groups_of_valid_subwords,
                        key=lambda group: sum(freq_dict[subword] for subword in group),
                    )
                    tokens = sorted_groups_of_valid_subwords[-1]
                    for token in tokens:
                        output_tokens.append(str(token))
        return output_tokens

    # https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/tokenization.py#L308
    def _tokenize_from_dict(
        self, text, freq_dict, use_cache, max_cache_size, max_word_size=20
    ):
        """Tokenize using frequency based approach given a dictionary

        Args:
            text (str): text to tokenize
            freq_dict (dict): a frequency dictionary
            use_cache (bool): whether to use caching 
            max_cache_size (int): max size for the caching dictionary
            max_word_size (int, optional): max word size. Defaults to 20.

        Returns:
            [type]: [description]
        """

        output_tokens = []
        cache = {}
        num_tokens = 0
        num_found_tokens = 0
        for token in text.split():
            num_tokens += 1
            chars = list(token)
            if len(chars) > max_word_size:
                output_tokens.append(self.unk)
                continue

            is_bad = False
            start = 0
            sub_tokens = []

            if use_cache:
                if token in cache:
                    output_tokens.extend(cache[token])
                    num_found_tokens += 1
                    continue
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in freq_dict:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end
            if is_bad:
                sub_tokens = [self.unk]
            output_tokens.extend(sub_tokens)
            if use_cache:
                if len(cache) < max_cache_size:
                    cache[token] = sub_tokens
        # print('Percentage of cached tokens  = ', num_found_tokens/num_tokens)
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
        for token in self.special_tokens:
            limited_tokens_frequency[token] = -1
        limited_tokens_frequency.update(
            {
                k: v
                for k, v in list(sorted_tokens_frequency.items())[
                    : self.vocab_size - len(limited_tokens_frequency)
                ]
            }
        )
        return limited_tokens_frequency

    def token_to_id(self, piece):
        """ Get tokens list

        Returns:
            list: tokens 
        """
        return list(self.vocab.keys()).index(piece)

    def id_to_token(self, id):
        """convert id to token

        Args:
            id (int): input id

        Returns:
            str: token
        """
        return list(self.vocab.keys())[id]

    def tokenize(self, text, use_cache=False, max_cache_size=1000):
        """tokenize

        Args:
            text (str): input text
            use_cache (bool, optional): speed up using caching. Defaults to False.
            max_cache_size (int, optional): max cacne size. Defaults to 1000.

        Returns:
            list: output list of tokens
        """
        output_tokens = self._tokenize_from_dict(
            text, self.vocab, use_cache, max_cache_size=max_cache_size
        )
        return output_tokens

    def _tokenize_word(self, text, remove_sow = True):
        """tokenize a single word

        Args:
            text (str): input text
            use_cache (bool, optional): speed up using caching. Defaults to False.
            max_cache_size (int, optional): max cacne size. Defaults to 1000.

        Returns:
            list: output list of tokens
        """
        output_tokens = self.tokenize(text)
        if remove_sow:
            return [token.replace(self.sow, "") for token in output_tokens]
        else:
            return output_tokens

    def detokenize(self, tokens):
        """ Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = " ".join(tokens).replace(f" {self.sow}", "")
        return detokenized

    def decode(self, encoded):
        """ Decode ids

        Args:
            encoded (list): list of ids to decode

        Returns:
            list: tokens
        """
        decoded = [self.id_to_token(id) for id in encoded]
        return decoded
    
    def decode_sentences(self, encoded):
        """ Decode list of lists of ids

        Args:
            encoded (list of list): list of list of ids to decode

        Returns:
            list: sentences
        """
        decoded = [[self.id_to_token(id) for id in ids if id not in [0, 3, 4]] for ids in encoded]
        decoded = [self.detokenize(tokens) for tokens in decoded]
        return decoded

    def encode(self, text):
        """ Convert string to a list of ids

        Args:
            text (str): input string

        Returns:
            list: list of ids
        """
        tokens = self.tokenize(text)
        encoded = [self.token_to_id(token) for token in tokens]
        return encoded

    def _encode_word(self, word, remove_sow = False):
        """ convert a word to ids

        Args:
            text (str): input string

        Returns:
            list: list of ids
        """
        tokens = self._tokenize_word(word, remove_sow=remove_sow)
        encoded = [self.token_to_id(token) for token in tokens]
        return encoded

    def pad_ids(self, ids, length = 0):
        """pad a set of ids to a specific length

        Args:
            ids (list): list of ids
            length (int, optional): Size to pad to. Defaults to 0.

        Returns:
            list: padded ids.
        """
        pad_id = self.token_to_id(self.pad)
        if length <= len(ids):
            return ids
        else:
            while len(ids) <= length:
                ids.append(pad_id)
        return ids

    def encode_sentences(self, sentences, add_boundry = False, out_length=None):
        """
        Encode a list of sentences using the trained model

        Args:
            sentences (list): list of sentences
            add_boundry (boolean): whether to add sos and eos. 
            out_length (int, optional): specify the max length of encodings. Defaults to 100.

        Returns:
            [list]: array of encodings
        """
        encodings = []
        if add_boundry:
            boundries = [SOS, EOS]

        max_length = 0
        pbar = tqdm(total=len(sentences)*2) 
        for sent in sentences:
            encoded = self.encode(sent)
            if add_boundry:
                encoded = [self.token_to_id(boundries[0])] + encoded + [self.token_to_id(boundries[1])]
            if len(encoded) > max_length:
                max_length = len(encoded)
            encodings.append(encoded)
            pbar.update(1)

        if out_length:
            max_length = max(max_length, out_length)

        pad_id = self.token_to_id(self.pad)
        for i in range(len(encodings)):
            encodings[i] = self.pad_ids(encodings[i], max_length)[:out_length]
            if encodings[i][-1] != pad_id and add_boundry:
                encodings[i][-1] = self.token_to_id(boundries[1])
            pbar.update(1)
        pbar.close()
        return encodings

    def load(self, file_path, name = 'tok'):
        """Load a saved model as a frequency dictionary

        Args:
            file_path (str): file path of the dictionary
        """
        with open(f'{file_path}/{name}.model', 'rb') as handle:
            self.vocab = pickle.load(handle)

    def save(self, file_path, name = 'tok'):
        """Save a model as a freqency dictionary

        Args:
            file_path (str): file path to save the model
            name (str): name of the file 
        """
        assert self.vocab
        os.makedirs(file_path, exist_ok=True)

        with open(f'{file_path}/{name}.model', 'wb') as handle:
            pickle.dump(self.vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __str__(self):
        return f"{self.__class__.__name__}"

    def calculate_compression_factor(self, text, normalized=True):
        factor = 0
        words = text.split()
        for word in words:
            tokenized = self.tokenize(word)
            factor += (
                len(word) + 1
                if self.unk in tokenized
                else len(tokenized)
            )
        if normalized:
            normalized_factor = factor / (
                sum(len(word) + 1 for word in words)
            )
            return normalized_factor
        return factor

