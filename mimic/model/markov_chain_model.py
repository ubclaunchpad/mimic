"""Markov chain model class."""

import random
from mimic.model.model import Model
from collections import defaultdict
import logging
import os
import pickle


class MarkovChainModel(Model):
    """A type of model."""

    def __init__(self, stateLength=1, predictionLength=50):
        """
        Constructor.

        Takes an int stateLength as an argument
        and instantiates a model.
        """
        self.order = stateLength
        self.groupSize = stateLength + 1
        self.dict = defaultdict(list)
        self.predictionLength = predictionLength
        self.data = None
        self.dump = None
        logging.info('Markov Model instantiated')

    def learn(self, data):
        """
        Learn method.

        Takes in a list of words as an argument
        and constructs a dictionary based
        on stateLength provided by the user.
        """
        logging.info('Learning...')
        self.data = data.split()

        for i in range(0, len(self.data) - self.groupSize):
            key = tuple(self.data[i: i + self.order])
            value = self.data[i + self.order]
            self.dict[key].append(value)

        logging.info('Finished Learning')
        self.dump = (self.order, self.dict, self.data)

    def predict(self, seed_text, pred_len):
        """
        Predict method.

        Uses the generated dictionary to create a
        sentence of specified length.
        """
        logging.info('Predicting')

        self.predictionLength = pred_len
        if seed_text is None:
            index = random.randint(0, len(self.data) - self.order)
        else:
            try:
                index = self.data.index(seed_text)
            except Exception:
                index = random.randint(0, len(self.data) - self.order)

        result = self.data[index: index + self.order]

        for _ in range(self.predictionLength):
            state = tuple(result[len(result) - self.order:])
            next = random.choice(self.dict[state])
            result.append(next)

        logging.info('Text successfully generated.')
        logging.info('--------')
        return " ".join(result)
        # return " ".join(result[self.order:])

    def save_trained_model(self, path, filename):
        """Save model as a pickle file."""
        output_path = os.path.join(path, filename + ".pickle")
        pickle_out = open(output_path, "wb")
        pickle.dump(self.dump, pickle_out)
        pickle_out.close()

    def load_pretrained_model(self, input_path, text=None):
        """Load pickle file and reassigns values."""
        try:
            pickle_in = open(input_path, "rb")
            import_dump = pickle.load(pickle_in)
            pickle_in.close()
            self.order, self.dict, self.data = import_dump
            self.groupSize = self.order + 1

        except (ImportError, ValueError) as e:
            logging.error(e)
            return False
