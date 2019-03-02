"""Core text translator module."""
from mimic.model.translation_model import TranslationModel


class TextTranslator:
    """
    Core text translator class.

    User-facing class that offers loading of bilingual dataset for training
    and predition functionalities.
    """

    def __init__(self):
        """Initialize a TextTranslator."""
        self.model = TranslationModel()

    def load_bilingual_text_file(self, bilingual_file_path):
        """
        Load training dataset for consumption by the model.

        The dataset is a pkl file of lines of phrases in source language
        in 1st column and in target laungauge in 2nd column, seperated by a
        tab.

        Example:
        Hi.	Hallo!
        Hi.	Grüß Gott!
        Run!	Lauf!
        Wow!	Potzdonner!
        Wow!	Donnerwetter!
        """
        raise NotImplementedError

    def translate_text(self):
        """Translate text to the target language of the training dataset."""
        raise NotImplementedError
