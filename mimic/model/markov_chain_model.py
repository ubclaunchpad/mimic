"""Markov chain model class."""

import random
from mimic.model.model import Model

class MarkovChainModel(Model):
    """A type of model"""

    def __init__ (self, stateLength):
        self.order = stateLength
        self.groupSize = stateLength + 1
        self.dict = {}
        self.data = None

    def learn(self, data):
        """Takes in a list of words as an argument
        and constructs a dictionary based
        on stateLength provided by the user"""
        print('Learning...')
        self.data = data.split()

        for i in range(0,len(data) - self.groupSize):
            key = tuple(data[i: i +self.order])
            value = data[i+ self.order]

            if key in self.dict:
                self.dict[key].append(value)
            else:
                self.dict[key] = [value]

        print('Finished Learning')
        print('--------')


    def predict(self, length):
        """Uses the generated dictionary to create a sentence of specified length"""
        index = random.randint(0,len(self.data) - self.order)
        result = self.data[index: index + self.order]

        for i in range(length):
            state = tuple(result[len(result) - self.order:])
            next = random.choice(self.dict[state])
            result.append(next)

        return " ".join(result[self.order:])

