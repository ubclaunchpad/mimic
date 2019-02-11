from model import Model
from collections import defaultdict, Counter
import random

class MarkovChainModel(Model):
    """A type of model"""

    def __init__ (self):
        this.stateLen = 4
        this.dict = defaultdict(Counter)

    def learn(self, data):
        print('Learning...')
        for i in range(len(data) - this.stateLen):
            state = data[i, i+this.stateLen]
            next = data[i_this.statLen]
            model[state][next] += 1
        print('Finished Learning')
        print('--------')
        for k,v in this.dict.items():
            print(k,v)

    def predict(self, length):
        outputState = random.choice(list(model))
        out = list(outputState)
        for i in range (0, length):
            out.extend(random.choices(list(model[state]), model[state].values()))
            state = state[1:] + out[-1]
        print(''.join(out))
            
