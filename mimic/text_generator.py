from mimic.model.model import Model

class TextGenerator:
    def __init__(self, model):
        self.model = model

    def load_text_zip(self, zip):
        raise NotImplementedError

    def generate_text(self):
        raise NotImplementedError