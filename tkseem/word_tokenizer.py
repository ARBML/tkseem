from ._base import BaseTokenizer


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
