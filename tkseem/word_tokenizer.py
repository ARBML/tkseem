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

    # Why do we have two versions of this method?
    def _tokens_list(self):
        """ Get tokens list

        Returns:
            list: tokens 
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
        detokenized = " ".join(tokens)
        return detokenized
