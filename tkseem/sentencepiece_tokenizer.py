import io

import sentencepiece as spm

from ._base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    """Sentencepiece based tokenization."""

    def train(self, file_path, model_type="bpe", **kwargs):
        """Train using sentence piece

        Args:
            file_path (str): file to train
            model_type (str, optional): train using sp. Defaults to "bpe".
            kwargs: additional arguments to pass to the SentencePieceTrainer. See https://github.com/google/sentencepiece/blob/master/doc/options.md
        """
        print("Training SentencePiece ...")
        self.model = io.BytesIO()

        if kwargs.get("vocab_size"):
            print(
                f"WARNING: Vocab size is being overwritten to {kwargs.get('vocab_size')}"
            )
            self.vocab_size = kwargs.get("vocab_size")

        if kwargs.get("special_tokens"):
            print(
                f"WARNING: Special tokens are being overwritten to {kwargs.get('special_tokens')}"
            )
            self.special_tokens = kwargs.get("special_tokens")

        spm.SentencePieceTrainer.train(
            input=file_path,
            model_writer=self.model,
            vocab_size=self.vocab_size,
            model_type=model_type,
            character_coverage=kwargs.get("character_coverage", 1.0),
            max_sentencepiece_length=kwargs.get("max_sentencepiece_length", 16),
            unk_id=0,
            pad_id=1,
            bos_id=-1,
            eos_id=-1,
            user_defined_symbols=self.special_tokens,
            train_extremely_large_corpus=kwargs.get(
                "train_extremely_large_corpus", False
            ),
            normalization_rule_name=kwargs.get("normalization_rule_name", "identity"),
        )
        self.sp = spm.SentencePieceProcessor(model_proto=self.model.getvalue())
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
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(file_path)

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
        """Convert string to a list of ids

        Args:
            text (str): input string

        Returns:
            list: list of ids
        """
        return self.sp.encode(text, out_type=int)

    def decode(self, encoded):
        """Decode ids

        Args:
            encoded (list): list of ids to decode

        Returns:
            list: tokens
        """
        return self.sp.id_to_piece(encoded)

    def detokenize(self, tokens):
        """Convert tokens to a string

        Args:
            tokens (list): list of tokens

        Returns:
            str: detokenized string
        """
        return "".join(tokens).replace("▁", " ")
