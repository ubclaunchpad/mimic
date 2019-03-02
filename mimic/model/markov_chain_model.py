"""Markov chain model class."""

import random
from mimic.model.model import Model
from collections import defaultdict
import logging
import pickle


class MarkovChainModel(Model):
    """A type of model."""

    def __init__(self, stateLength):
        """
        Constructor.

        Takes an int stateLength as an argument
        and instantiates a model.
        """
        self.order = stateLength
        self.groupSize = stateLength + 1
        self.dict = defaultdict(list)
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
        logging.info('--------')
        self.dump = (self.order, self.dict, self.data)


    def predict(self, length):
        """
        Predict method.

        Uses the generated dictionary to create a
        sentence of specified length.
        """
        logging.info('Predicting')
        index = random.randint(0, len(self.data) - self.order)
        result = self.data[index: index + self.order]

        for _ in range(length):
            state = tuple(result[len(result) - self.order:])
            next = random.choice(self.dict[state])
            result.append(next)

        return " ".join(result[self.order:])

    def save(self, path, filename):
        """Save model as a pickle file."""
        pickle_out = open(path+filename+".pickle", "wb")
        pickle.dump(self.dump, pickle_out)
        pickle_out.close()

    def load(self, path):
        """Load pickle file and reassigns values."""
        pickle_in = open(path + ".pickle", "rb")
        importDump = pickle.load(pickle_in)
        pickle_in.close()
        self.order, self.dict, self.data = importDump
        self.groupSize = self.order + 1
