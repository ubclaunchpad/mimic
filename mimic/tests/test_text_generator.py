import unittest
from mimic.model.model import *
from mimic.text_generator import TextGenerator


class TestTextGenerator(unittest.TestCase):

    def setUp(self):
        self.textGenerator = TextGenerator(Model())

    def test_load_text_zip(self):
        pass
        # self.textGenerator.load_text_zip("zip")
        # assert something

    def test_generate(self):
        pass
        # self.textGenerator.generate_text()
        # assert something else
