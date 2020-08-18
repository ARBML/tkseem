import unittest

import tkseem as tk


class TokenizersTestUnit(unittest.TestCase):
    sample_text = "مرحبا أيها الأصدقاء"
    token = "نص"
    chars = False
    tokenizer = None

    def print_string(self, method_name, output, method_arguments=None):
        return f'{self.tokenizer} method {method_name}() output {f"on arguments: [{method_arguments}]" if method_arguments else ""} is: {output}'

    def test_tokenize(self):
        tokenized = self.tokenizer.tokenize(self.sample_text)
        print(self.print_string("tokenize", tokenized, method_arguments="sample_text"))
        return self.assertIsNotNone(tokenized)

    def test_detokenize(self):
        tokenized = self.tokenizer.tokenize(self.sample_text)
        detokenized = self.tokenizer.detokenize(tokenized)
        print(
            self.print_string(
                "detokenize", detokenized, method_arguments="tokenized sample_text"
            )
        )
        return self.assertIsNotNone(detokenized)

    def test_token_to_id(self):
        token = self.token if not self.chars else self.token[0]
        token_id = self.tokenizer.token_to_id(token)
        print(self.print_string("token_to_id", token_id, method_arguments=token))
        return self.assertIsNotNone(token_id)

    def test_id_to_token(self):
        token = self.token if not self.chars else self.token[0]
        token_id = self.tokenizer.token_to_id(token)
        matched_token = self.tokenizer.id_to_token(token_id)
        print(
            self.print_string(
                "id_to_token", matched_token, method_arguments=f"'{token}' id"
            )
        )
        return self.assertEqual(matched_token, token)

    def test_encode(self):
        encoded = self.tokenizer.encode(self.sample_text)
        print(self.print_string("encode", encoded, method_arguments="sample_text"))
        return self.assertIsNotNone(encoded)

    def test_decode(self):
        encoded = self.tokenizer.encode(self.sample_text)
        decoded = self.tokenizer.decode(encoded)
        print(
            self.print_string("decode", decoded, method_arguments="encoded sample_text")
        )
        return self.assertIsNotNone(decoded)


for tokenizer in (
    tk.SentencePieceTokenizer(),
    tk.WordTokenizer(),
    tk.MorphologicalTokenizer(),
    tk.CharacterTokenizer(),
    tk.DisjointLetterTokenizer(),
    tk.RandomTokenizer(),
):
    try:
        tokenizer.train("tasks/samples/data.txt")
    except TypeError as type_error:
        print(f"{tokenizer} does not need file_path to train")
        tokenizer.train()
    if isinstance(tokenizer, tk.CharacterTokenizer):
        TokenizersTestUnit.chars = True
    TokenizersTestUnit.tokenizer = tokenizer
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
