class Model:
    """an abstract class representing a general model
    all models will share these behaviors"""

    def __init__(self):
        """initializes the model"""
        pass

    def learn(self, text):
        """trains the model"""
        raise NotImplementedError

    def predict(self):
        """generates output"""
        raise NotImplementedError