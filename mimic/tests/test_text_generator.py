import unittest
from mimic.model.model import *
from mimic.text_generator import TextGenerator


class TestTextGenerator(unittest.TestCase):

    def setUp(self):
        self.textGenerator = TextGenerator(Model())

    def test_load_text_zip(self):
        text = self.textGenerator.load_text_zip("../../data/test.zip")
        expected = "Test 3 is also done now. This is a second test.\n" \
                   "Test 2 is done. This a test file.\n" \
                   "Test 1 done."
        assert text == expected

    def test_generate(self):
        pass
        # self.textGenerator.generate_text()
        # assert something else
