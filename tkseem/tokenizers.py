import functools
import io
import itertools
import mmap
import operator
import os
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sentencepiece as spm
from farasa.segmenter import FarasaSegmenter
from tqdm import tqdm

from .util import clean_data, normalize_data, split_on_binary


class BaseTokenizer:
    """
    Base Tokenizer that implements the basic functionalities of a tokenizer
    """

    def __init__(
        self, unk_token="<UNK>", pad="<PAD>", vocab_size=10000, special_tokens=[],
    ):
        """Constructor

        Args:
            unk_token (str, optional): reserved token for unknowns. Defaults to "<UNK>".
            pad (str, optional): reserved token for padding. Defaults to "<PAD>".
            max_tokens (int, optional): max number of vocabulary. Defaults to 10000.
        """
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad = pad
        self.special_tokens = special_tokens

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

    def _write_data(self, path, data):
        """
        Write the string data to a path

        Args:
            file_path (str): the directory of the data to read
        
        """
        # TOCHECK: I think this code will break if the path does not exist.
        open(path, "w").write(data)

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

    def _tokenize_from_dict(self, text, freq_dict, cache=False, max_size=20):
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
                output_tokens.append(self.unk_token)
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
        limited_tokens_frequency[self.pad] = -1
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
        """ Get tokens list

        Returns:
            list: tokens 
        """
        return list(self.vocab.keys())[id]

    def encode(self, text):
        """
        Convert text to ids 
        """
        raise NotImplementedError

    def decode(self, encoded):
        """
        Convert ids to string
        """
        return NotImplementedError

    def tokenize(self, text):
        """
        Convert text to tokens
        """
        raise NotImplementedError

    def detokenize(self, tokens):
        """
        Convert tokens to text
        """
        raise NotImplementedError

    def encode_and_save(self):
        """
        Encode all the files then save as numpy
        """
        Path("data/encoded").mkdir(parents=True, exist_ok=True)
        for file_path in os.listdir("data/raw/"):
            ids = self.encode(open(f"data/raw/{file_path}", "r").read())
            np.save(f"data/encoded/{file_path[:-4]}.npy", ids)

    def encode_sentences(self, sentences, boundries=("", ""), out_length=None):
        """
        Encode a list of sentences using the trained model

        Args:
            sentences (list): list of sentences
            out_length (int, optional): specify the max length of encodings. Defaults to 100.

        Returns:
            [np.array]: numpy array of encodings
        """
        encodings = []
        for sent in sentences:
            encoded = self.encode(boundries[0] + " " + sent + " " + boundries[1])
            encodings.append(encoded)

        pad_id = self.encode(self.pad)[0]

        # pad to equal size from https://stackoverflow.com/a/38619333
        encodings = np.array(
            list(itertools.zip_longest(*encodings, fillvalue=pad_id))
        ).T

        # increase pad if necessary
        if not (out_length is None):
            if out_length > encodings.shape[1]:
                encodings = np.pad(
                    encodings,
                    [(0, 0), (0, out_length)],
                    constant_values=pad_id,
                    mode="constant",
                )
        encodings = encodings[..., :out_length]

        return encodings

    def __str__(self):
        return f"{self.__class__.__name__} tokenizer"


class WordTokenizer(BaseTokenizer):
    """
    White space based tokenization 
    """

    tokens_frequency = None

    def train(self, file_path):
        """Train using words' frequency

        Args:
            file_path (str): file path for daaset
        """

        print("Training WordTokenizer ...")
        self.vocab = self._truncate_dict(self._get_tokens_frequency(file_path))
        self.vocab_size = len(self.vocab)

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

    def decode(self, encoded):
        """ Decode ids

        Args:
            encoded (list): list of ids to decode

        Returns:
            list: tokens
        """
        decoded = [self.id_to_token(id) for id in encoded]
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

    def detokenize(self, tokens):
        """ Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = " ".join(tokens)
        return detokenized


class SentencePieceTokenizer(BaseTokenizer):
    """ Sentencepiece based tokenization. 
    """

    def train(self, file_path, model_type="bpe"):
        """ Train using sentence piece

        Args:
            model_type (str, optional): train using sp. Defaults to "bpe".
        """
        print("Training SentencePiece ...")
        self.model = io.BytesIO()

        spm.SentencePieceTrainer.train(
            input=file_path,
            model_writer=self.model,
            vocab_size=self.vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            unk_id=0,
            pad_id=1,
            bos_id=-1,
            eos_id=-1,
            user_defined_symbols=self.special_tokens,
            normalization_rule_name="identity",
        )
        self.save_model("m.model")
        self.sp = spm.SentencePieceProcessor(model_file="m.model")
        self.vocab_size = self.sp.vocab_size()

    def tokenize(self, text):
        """Tokenize using the frequency dictionary 

        Args:
            text (str): input string

        Returns:
            list: generated tokens
        """
        return self.sp.encode(text, out_type=str)

    def load_model(self, file_path):
        """Load a saved sp model

        Args:
            file_path (str): file path of the trained model
        """
        sp = spm.SentencePieceProcessor()
        self.sp = sp.Load(file_path)

    def save_model(self, file_path):
        """Save a model as a freqency dictionary

        Args:
            file_path (str): file path to save the model
        """
        with open(file_path, "wb") as f:
            f.write(self.model.getvalue())

    def id_to_token(self, id):
        return self.sp.id_to_piece(int(id))

    def token_to_id(self, token):
        return self.sp.piece_to_id(token)

    def encode(self, text):
        """ Convert string to a list of ids

        Args:
            text (str): input string

        Returns:
            list: list of ids
        """
        return self.sp.encode(text, out_type=int)

    def decode(self, encoded):
        """ Decode ids

        Args:
            encoded (list): list of ids to decode

        Returns:
            list: tokens
        """
        return self.sp.id_to_piece(encoded)

    def detokenize(self, tokens):
        """ Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        return "".join(tokens).replace("▁", " ")


class AutoTokenizer(BaseTokenizer):
    """ Auto tokenization using a saved dictionary 
    """

    def train(self, vocab_path):
        """Train the tokenizer using the given dictionary

        Args:
            vocab_path (str): dictionary to use for training 
        """
        print("Training AutoTokenizer ...")
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

    def decode(self, encoded):
        """ Decode ids

        Args:
            encoded (list): list of ids to decode

        Returns:
            list: tokens
        """
        decoded = [self.id_to_token(id) for id in encoded]
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

    def detokenize(self, tokens):
        """ Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = "".join(tokens).replace("##", "")
        return detokenized


class RandomTokenizer(BaseTokenizer):
    """ Randomized based tokenization"""

    def train(self, file_path):
        """Train data using randomly splitted subwords 
        """
        print("Training RandomTokenizer ...")
        text = open(file_path, "r").read()
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

    def tokenize(self, text):
        """Tokenize using the frequency dictionary 

        Args:
            text (str): input string

        Returns:
            list: generated tokens
        """
        output_tokens = self._tokenize_from_dict(text, self.vocab)
        return output_tokens

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

    def decode(self, encoded):
        """ Decode ids

        Args:
            encoded (list): list of ids to decode

        Returns:
            list: tokens
        """
        decoded = [self.id_to_token(id) for id in encoded]
        return decoded

    def encode(self, text):
        """ Convert string to a list of ids

        Args:
            text (str): input string

        Returns:
            list: list of ids
        """
        # TOCKECK: Why not to put this in the base tokenizer as a default behaviour?
        tokens = self.tokenize(text)
        encoded = [self.token_to_id(token) for token in tokens]
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


class DisjointLetterTokenizer(BaseTokenizer):
    """ Disjoint Letters based tokenization 
    """

    def train(self, file_path):
        """Train data using disjoint letters
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

    def tokenize(self, text):
        """Tokenize using the frequency dictionary 

        Args:
            text (str): input string

        Returns:
            list: generated tokens
        """
        output_tokens = self._tokenize_from_dict(text, self.vocab)
        return output_tokens

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

    def decode(self, encoded):
        """ Decode ids

        Args:
            encoded (list): list of ids to decode

        Returns:
            list: tokens
        """
        decoded = [self.id_to_token(id) for id in encoded]
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

    def detokenize(self, tokens):
        """ Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = "".join(tokens).replace("##", "")
        return detokenized


class CharacterTokenizer(BaseTokenizer):
    """ Character based tokenization 
    """

    def train(self, file_path):
        """Train data using characters 
        """
        print("Training CharacterTokenizer ...")
        rx = re.compile(r"\B(.)")

        text = open(file_path, "r").read()
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
                output_tokens.append(self.unk_token)
        return output_tokens

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

    def decode(self, encoded):
        """ Decode ids

        Args:
            encoded (list): list of ids to decode

        Returns:
            list: tokens
        """
        decoded = [self.id_to_token(id) for id in encoded]
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

    def detokenize(self, tokens):
        """ Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = "".join(tokens).replace("##", "")
        return detokenized
class MorphologicalTokenizer(BaseTokenizer):
    """ Use Morphology to tokenize the text 
    """

    def train(self, file_path):
        """Use a default dictionary for training"""
        print("Training MorphologicalTokenizer ...")
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

    def decode(self, encoded):
        """ Decode ids

        Args:
            encoded (list): list of ids to decode

        Returns:
            list: tokens
        """
        decoded = [self.id_to_token(id) for id in encoded]
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

    def detokenize(self, tokens):
        """ Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        detokenized = "".join(tokens).replace("##", "")
        return detokenized
