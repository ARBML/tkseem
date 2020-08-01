from .__base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    """ Character based tokenization 
    """

    def train(self):
        """Train data using characters 
        """
        print("Training CharacterTokenizer ...")
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
