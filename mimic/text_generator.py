"""Core text generator module."""
from mimic.model.model import Model


class TextGenerator:
    """
    Core text generation class.

    User-facing class that abstracts away text processing details and
    interacts with models.
    """

    def __init__(self, model):
        """Initialize a TextGenerator with the given model type."""
        self.model = model

    def load_text_zip(self, zip):
        """Prepare a text file for consumption by the model."""
        raise NotImplementedError

    def generate_text(self):
        """Generate textual output based on training data."""
        raise NotImplementedError
