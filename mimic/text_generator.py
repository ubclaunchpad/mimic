"""Core text generator module."""
from mimic.model.model import Model
from mimic.model.markov_chain_model import MarkovChainModel
from mimic.model.lstm_model import LSTMModel
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

    def train_from_zip(self, zip_file_path):
        """Load the zip and initiate training by the model."""
        text = load_text_zip(zip_file_path)
        model.learn(text)

    def load_text_zip(self, zip_file_path):
        """Prepare a text file for consumption by the model."""
        files = []

        with zipfile.ZipFile(zip_file_path, "r") as archive:
            for name in archive.namelist():
                if name.endswith(".txt"):
                    files.append(archive.read(name).decode('utf-8'))
            dir_name = os.path.dirname(zip_file_path)
            text_string = " ".join(files)
        return text_string

    def train_from_zip(self, zip_file_path):
        """Load the zip and initiate training by the model."""
        text = self.load_text_zip(zip_file_path)
        self.model.learn(text)

    def load_pretrained_model(self, filepath, text=None):
        """Load the pretrained model."""
        self.model.load_pretrained_model(filepath, text)

    def save_trained_model(self, dir_path, filename):
        """Save the trained model."""
        self.model.save_trained_model(dir_path, filename)

    def generate_text(self, seed_text=None, pred_len=50):
        """Generate textual output based on training data."""
        return self.model.predict(seed_text, pred_len)
