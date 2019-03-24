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

    def load_pretrained_model(self, filepath):
        """
        Load a pretrained model file from a filepath.

        Loads model and return true if loaded or false if an error occured.
        """
        raise NotImplementedError

    def save_trained_model(self, dir_path, filename):
        """Save the trained model to a file and return file path."""
        raise NotImplementedError
