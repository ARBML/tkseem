import io
import os
import sentencepiece as spm
from ._base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    """ Sentencepiece based tokenization. 
    """

    def __init__(
        self, vocab_size=10000,
    ):
        super(SentencePieceTokenizer, self).__init__(vocab_size=vocab_size)
        self.name = "SentencePieceTokenizer"
        self.sow = '▁'        

    def train(self, file_path, model_type="bpe"):
        """ Train using sentence piece

        Args:
            file_path (str): file to train 
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
            unk_id=self.unk_idx,
            pad_id=self.pad_idx,
            bos_id=self.sos_idx,
            eos_id=self.eos_idx,
            normalization_rule_name="identity",
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

    def load(self, file_path, name = 'tok'):
        """Load a saved sp model

        Args:
            file_path (str): file path of the trained model
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(f'{file_path}/{name}.model')

    def save(self, file_path, name = 'tok'):
        """Save a model as a freqency dictionary

        Args:
            file_path (str): file path to save the model
        """
        os.makedirs(file_path, exist_ok=True)
        with open(f'{file_path}/{name}.model', "wb") as f:
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
        return "".join(tokens).replace(f"{self.sow}", " ").strip()
