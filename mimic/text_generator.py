from mimic.model.model import Model
import zipfile, os

class TextGenerator:
    """a user facing class that abstracts away text processing details
     and interactions with the models"""

    def __init__(self, model):
        """initiates the a TextGenerator with a given model type"""
        self.model = model

    def load_text_zip(self, zip_file_path):
        """prepares a text file for consumption by the model"""

        with zipfile.ZipFile(zip_file_path, "r") as archive:
            files = [archive.read(name).decode('utf-8') for name in archive.namelist() if name.endswith(".txt")]
            dir_name = os.path.dirname(zip_file_path)
            text_string = " ".join(files)
            archive.extractall(dir_name)

        return text_string

    def generate_text(self):
        """generates textual output based on training data"""
        raise NotImplementedError

