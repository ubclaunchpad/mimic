"""Server."""
from flask import Flask
from mimic.text_generator import TextGenerator
from mimic.text_generator_factory import TextGeneratorFactory
app = Flask(__name__)


@app.route('/<model>/<string_length>', methods=['GET'])
def getText(model, string_length):
    """Prepare a text file for consumption by the model."""
    return 'We cant really do this until we have a pretrained model'
