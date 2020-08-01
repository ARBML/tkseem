from .__base import BaseTokenizer


class RandomTokenizer(BaseTokenizer):
    """ Randomized based tokenization 
    """

    def train(self):
        """Train data using randomly splitted subwords 
        """
        print("Training RandomTokenizer ...")
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
        # TOCKECK: Why not to put this in the base tokenizer as a default behaviour?
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


class DisjointLetterTokenizer(BaseTokenizer):
    """ Disjoint Letters based tokenization 
    """

    def train(self):
        """Train data using disjoint letters
        """
        print("Training DisjointLetterTokenizer ...")
        rx = re.compile(r"([اأإآءؤﻵﻹﻷدذرزو])")

        text = open("data/raw/train.txt", "r").read()
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
