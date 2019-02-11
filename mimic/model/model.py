"""Model superclass."""


class Model:
    """
    An abstract class representing a general model.

    All models will share these behaviors.
    """

    def __init__(self):
        """Initialize the model."""
        pass

    def learn(self, text):
        """Trains the model."""
        raise NotImplementedError

    def predict(self):
        """Generate output."""
        raise NotImplementedError
