"""Core text generator module."""
# from mimic.model.model import Model
# from mimic.model.markov_chain_model import MarkovChainModel
# from mimic.model.lstm_model import LSTMModel
import zipfile
import os

class TextGenerator:
    """
    Core text generation class.

    User-facing class that abstracts away text processing details and
    interacts with models.
    """

    def __init__(self, model):
        """Initialize a TextGenerator with the given model type."""
        self.model = model

    def load_text_zip(self, zip_file_path):
        """Prepare a text file for consumption by the model."""
        files = []

        with zipfile.ZipFile(zip_file_path, "r") as archive:
            for name in archive.namelist():
                if name.endswith(".txt"):
                    files.append(archive.read(name).decode('utf-8'))
            dir_name = os.path.dirname(zip_file_path)
            text_string = " ".join(files)
            archive.extractall(dir_name)

        self.data = text_string

    def generate_text(self):
        """Generate textual output based on training data."""

        self.model.learn(self.data)
        print(self.model.predict())
