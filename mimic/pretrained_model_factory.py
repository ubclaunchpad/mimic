"""Pretrained model factory."""
from mimic.model.markov_chain_model import MarkovChainModel
from mimic.model.lstm_model import LSTMModel
from mimic.text_generator import TextGenerator
import mimic.util as utils
import os
import logging


class PretrainedModelFactory:
    """
    Pretrained Model Class.

    Class that creates a text generator of a pretrained model chosen by user.
    """

    def __init__(self):
        """Initialize a TextGenerator with the given pretrained model type."""
        self.data_path = os.path.join(os.getcwd(), "data")
        self.pretrained_models_path = os.path.join(os.getcwd(), "data",
                                                   "pretrained_models")

    def create_pretrained_LSTM_trump_tweets_generator(self):
        """Use a pretrained LSTM model with Trump Tweets."""
        logging.info("Creating pretrained LSTM Trump tweets generator")

        self.model = TextGenerator(LSTMModel())
        text = self.model.load_text_zip(os.path.join(self.data_path,
                                        "trump_tweets.zip"))
        self.model.load_pretrained_model(os.path.join(
                                         self.pretrained_models_path,
                                         "trained_LSTM_trump_tweets.h5"),
                                         text)
        # self.generate_LSTM_inclass_variables(text)
        return self.model

    def create_pretrained_LSTM_shakespeare_text_generator(self):
        """Use a pretrained LSTM model with Shakespeare text."""
        logging.info("Creating pretrained LSTM Shakespeare text generator")

        self.model = TextGenerator(LSTMModel())
        text = self.model.load_text_zip(os.path.join(self.data_path,
                                        "clean_shakespeare.zip"))
        self.model.load_pretrained_model(os.path.join(
                                         self.pretrained_models_path,
                                         "trained_LSTM_clean_shakespeare.h5"),
                                         text)
        # self.generate_LSTM_inclass_variables(text)
        return self.model

    def create_pretrained_markov_chain_trump_tweets_generator(self):
        """Use a pretrained Markov Chain model with Trump tweets."""
        logging.info("Creating pretrained Markov Chain Trump tweets generator")

        self.model = TextGenerator(MarkovChainModel())
        self.model.load_pretrained_model(os.path.join(
                                 self.pretrained_models_path,
                                 "trained_markov_model_trump_tweets.pickle"))
        return self.model

    def create_pretrained_markov_chain_shakespeare_text_generator(self):
        """Use a pretrained Markov Chain model with Shakespeare text."""
        logging.info("Creating pretrained Markov Chain Shakespeare text"
                     "generator")

        self.model = TextGenerator(MarkovChainModel())
        self.model.load_pretrained_model(os.path.join(
                                 self.pretrained_models_path,
                                 "trained_markov_model_shakespeare.pickle"))
        return self.model
