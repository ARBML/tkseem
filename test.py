import unittest
from tkseem.util import remove_tashkeel
import tkseem as tk


class TestUnit(unittest.TestCase):
    def test_tashkeel(self):
        self.assertEqual(
            remove_tashkeel("مِكَرٍّ مِفَرٍّ مُقبِلٍ مُدبِرٍ مَعًا"),
            "مكر مفر مقبل مدبر معا",
            "Remove Tashkeel is not working",
        )


class TokenizersTestUnit(unittest.TestCase):
    sample_text = "مرحبا أيها الأصدقاء"
    token = "نص"
    tokenizer = None
    token_id = None

    def test_tokenize(self):
        tokenized = self.tokenizer.tokenize(self.sample_text)
        print(f"{self.tokenizer} tokenize() output:", tokenized)
        return self.assertIsNotNone(tokenized)

    def test_detokenize(self):
        tokenized = self.tokenizer.tokenize(self.sample_text)
        detokenized = self.tokenizer.detokenize(tokenized)
        print(
            f"{self.tokenizer} detokenize() output on the previously tokenized text:",
            detokenized,
        )
        return self.assertIsNotNone(detokenized)

    def test_token_to_id(self):
        self.token_id = self.tokenizer.token_to_id(self.token)
        print(f"{self.tokenizer} token_to_id() output:", self.token_id)
        return self.assertIsNotNone(self.token_id)


word_tokenizer = tk.WordTokenizer()
word_tokenizer.train("tasks/samples/data.txt")
TokenizersTestUnit.tokenizer = word_tokenizer
unittest.main(argv=["first-arg-is-ignored"], exit=False)

