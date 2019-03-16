"""text generator creator module."""
from mimic.model.model import Model
from mimic.model.markov_chain_model import MarkovChainModel
from mimic.model.lstm_model import LSTMModel
from mimic.text_generator import TextGenerator
import logging

# Number of words to "learn" from
DEFAULT_MARKOV_STATE_LENGTH = 5
# Number of words in each training sequence
DEFAULT_LSTM_SEQ_LEN = 100
# Number of words to generate
DEFAULT_OUTPUT_LEN = 50


class TextGeneratorFactory:
    """
    TextGeneratorFactory class.

    Class that creates TextGenerators with the choice of a model for the user.
    """

    def __init__(self):
        """Initialize a TextGenerator with the given model type."""
        logging.basicConfig(format='%(asctime)s %(message)s',
                            filename='textgenerator.log',
                            level=logging.DEBUG)

    def create_markov_chain_text_generator(self):
        """Create a TextGenerator using a markov chain model."""
        return TextGenerator(MarkovChainModel(DEFAULT_MARKOV_STATE_LENGTH,
                                              DEFAULT_OUTPUT_LEN))

    def create_LTSM_text_generator(self):
        """Create a TextGenerator using a LTSM model."""
        return TextGenerator(LSTMModel(DEFAULT_LSTM_SEQ_LEN,
                                       DEFAULT_OUTPUT_LEN))
