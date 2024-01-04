import io

import sentencepiece as spm

from ._base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    """Sentencepiece based tokenization."""

    def train(self, file_path, **kwargs):
        """Train using sentence piece

        Args:
            file_path (str): file to train
            kwargs: additional arguments to pass to the SentencePieceTrainer. See https://github.com/google/sentencepiece/blob/master/doc/options.md
        """
        print("Training SentencePiece ...")
        self.model = io.BytesIO()

        if kwargs.get("vocab_size"):
            print(
                f"WARNING: Vocab size is being overwritten to {kwargs.get('vocab_size')}"
            )
            self.vocab_size = kwargs.get("vocab_size")
            kwargs.pop("vocab_size")

        if kwargs.get("special_tokens"):
            print(
                f"WARNING: Special tokens are being overwritten to {kwargs.get('special_tokens')}"
            )
            self.special_tokens = kwargs.get("special_tokens")
            kwargs.pop("special_tokens")

        # Preserve default values from previous versions
        model_type = kwargs.get("model_type", "bpe")
        kwargs.pop("model_type")
        character_coverage = kwargs.get("character_coverage", 1.0)
        kwargs.pop("character_coverage")
        unk_id = kwargs.get("unk_id", 0)
        kwargs.pop("unk_id")
        pad_id = kwargs.get("pad_id", 1)
        kwargs.pop("pad_id")
        bos_id = kwargs.get("bos_id", -1)
        kwargs.pop("bos_id")
        eos_id = kwargs.get("eos_id", -1)
        kwargs.pop("eos_id")
        normalization_rule_name = kwargs.get("normalization_rule_name", "identity")
        kwargs.pop("normalization_rule_name")

        spm.SentencePieceTrainer.train(
            input=file_path,
            model_writer=self.model,
            vocab_size=self.vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            unk_id=unk_id,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            user_defined_symbols=self.special_tokens,
            normalization_rule_name=normalization_rule_name,
            **kwargs,
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
