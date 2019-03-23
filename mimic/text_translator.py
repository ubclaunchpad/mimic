"""Core text translator module."""
import gzip
from mimic.model.translation_model import TranslationModel
from pickle import load


class TextTranslator:
    """
    Core text translator class.

    User-facing class that offers loading of bilingual dataset for training
    and predition functionalities.
    """

    def __init__(self):
        """Initialize a TextTranslator."""
        self.model = TranslationModel()

    def _load_bilingual_text_file(self, bilingual_file_path):
        """
        Load training dataset for consumption by the model.

        The dataset is a gz file of rows of text in source language
        in 1st column and in target laungauge in 2nd column.

        Example:
        Hi.	    Hallo!
        Hi.	    Grüß Gott!
        Run!	Lauf!
        Wow!	Potzdonner!
        Wow!	Donnerwetter!
        """
        return load(gzip.open(bilingual_file_path, 'rb'))

    
    def train_from_file(self, bilingual_file_path):
        bilingual_text = self._load_bilingual_text_file(bilingual_file_path)
        self.model.learn(bilingual_text)

    
    def translate_text(self, text):
        """Translate text to the target language of the training dataset."""
        return self.model.predict(text)
    

    def load_pretrained_model(self, filepath):
        """Load the pretrained model."""
        raise NotImplementedError

    def save_trained_model(self, dir_path, filename):
        """Save the trained model."""
        raise NotImplementedError