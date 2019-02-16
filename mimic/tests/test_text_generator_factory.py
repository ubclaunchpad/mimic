import unittest
from mimic.text_generator import TextGenerator
from mimic.text_generator_factory import TextGeneratorFactory


class TestTextGeneratorFactory(unittest.TestCase):

    def setUp(self):
        self.textGeneratorFactory = TextGeneratorFactory()

    def test_create_markov_chain_text_generator(self):
        assert(isinstance(self.textGeneratorFactory.create_markov_chain_text_generator(), TextGenerator))

    def test_create_LTSM_text_generator(self):
        # assert(isinstance(self.textGeneratorFactory.create_LTSM_text_generator(), TextGenerator))
        pass
