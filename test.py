import tokenizers
from utils import *
import unittest

class TestUnit(unittest.TestCase):
    def test_tashkeel(self):
        self.assertEqual(remove_tashkeel("مِكَرٍّ مِفَرٍّ مُقبِلٍ مُدبِرٍ مَعًا")
                        , "مكر مفر مقبل مدبر معا", "Remove Tashkeel is not working")

tokenizer = tokenizers.FrequencyTokenizer(normalize = False,
                                         clean = True, 
                                         segment = False)
tokenizer.read_data('samples/data.txt')
tokenizer.train()

unittest.main(argv=['first-arg-is-ignored'], exit=False)
