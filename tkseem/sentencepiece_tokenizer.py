import io
import numpy as np
import sentencepiece as spm
from .__base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    """ Sentencepiece based tokenization. 
    """

    def train(self, model_type="bpe"):
        """ Train using sentence piece

        Args:
            model_type (str, optional): train using sp. Defaults to "bpe".
        """
        print("Training SentencePiece...")
        self._check_train_data_path()
        self.model = io.BytesIO()
        spm.SentencePieceTrainer.train(
            input="data/raw/train.txt",
            model_writer=self.model,
            vocab_size=self.vocab_size,
            model_type=model_type,
            character_coverage=1.0,
            unk_id=0,
            pad_id=1,
            bos_id=-1,
            eos_id=-1,
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

    def encode_sentences(self, sentences, max_length=20):
        """Encode a list of sentences using the trained model

        Args:
            sentences (list): list of sentences
            max_length (int, optional): specify the max length of encodings. Defaults to 100.

        Returns:
            [np.array]: list of encoded sentences
        """
        sparse_encodings = self.sp.encode(sentences, out_type=int)
        encodings = []
        for encoding in sparse_encodings:
            curr_encoding = []
            for i in range(max_length):
                if i >= len(encoding):
                    curr_encoding.append(self.sp.pad_id())
                else:
                    curr_encoding.append(encoding[i])
            encodings.append(curr_encoding)
        return np.array(encodings)
