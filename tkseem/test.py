from .util import remove_tashkeel
import unittest

class TestUnit(unittest.TestCase):
    def test_tashkeel(self):
        self.assertEqual(remove_tashkeel("مِكَرٍّ مِفَرٍّ مُقبِلٍ مُدبِرٍ مَعًا")
                        , "مكر مفر مقبل مدبر معا", "Remove Tashkeel is not working")

unittest.main(argv=['first-arg-is-ignored'], exit=False)
