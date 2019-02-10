from mimic.model.model import Model

class TextGenerator:
    """a user facing class that abstracts away text processing details
     and interactions with the models"""

    def __init__(self, model):
        """initiates the a TextGenerator with a given model type"""
        self.model = model

    def load_text_zip(self, zip):
        """prepares a text file for consumption by the model"""
        raise NotImplementedError

    def generate_text(self):
        """generates textual output based on training data"""
        raise NotImplementedError