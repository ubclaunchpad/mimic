"""text generator creator module."""
from mimic.model.model import Model
from mimic.model.markov_chain_model import MarkovChainModel
from mimic.model.lstm_model import LSTMModel
from mimic.text_generator import TextGenerator
import logging

# Number of words to "learn" from
MARKOV_STATE_LENGTH = 5
# Number of words in each training sequence
LSTM_SEQ_LEN = 100
# Number of words to generate
OUTPUT_LEN = 50


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
        return TextGenerator(MarkovChainModel(MARKOV_STATE_LENGTH,
                                              OUTPUT_LEN))

    def create_LSTM_text_generator(self):
        """Create a TextGenerator using a LTSM model."""
        return TextGenerator(LSTMModel(LSTM_SEQ_LEN,
                                       OUTPUT_LEN))
