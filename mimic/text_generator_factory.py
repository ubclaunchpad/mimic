"""text generator creator module."""
from mimic.model.model import Model
from mimic.model.markov_chain_model import MarkovChainModel
from mimic.text_generator import TextGenerator


class TextGeneratorFactory:
    """
    TextGeneratorFactory class.

    Class that creates TextGenerators with the choice of a model for the user.
    """

    def __init__(self):
        """Initialize a TextGenerator with the given model type."""
        logging.basicConfig(format='%(asctime)s %(message)s', filename='textGenerator.log', level=logging.DEBUG)

    def create_markov_chain_text_generator(self):
        """Create a TextGenerator using a markov chain model."""
        return TextGenerator(MarkovChainModel())

    def create_LTSM_text_generator(self):
        """Create a TextGenerator using a LTSM model."""
        raise NotImplementedError
