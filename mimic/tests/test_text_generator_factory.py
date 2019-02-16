import unittest
from mimic.text_generator import TextGenerator
from mimic.text_generator_factory import TextGeneratorFactory


class TestTextGeneratorFactory(unittest.TestCase):

    def setUp(self):
        self.textGeneratorFactory = TextGeneratorFactory()

    def test_create_markov_chain_text_generator(self):
        factory = self.textGeneratorFactory
        testGenerator = factory.create_markov_chain_text_generator()
        assert(isinstance(testGenerator, TextGenerator))

    def test_create_LTSM_text_generator(self):
        # factory = self.textGeneratorFactory
        # testGenerator = factory.create_LTSM_text_generator()
        # assert(isinstance(testGenerator, TextGenerator))
        pass
