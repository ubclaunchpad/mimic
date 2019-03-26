"""Server."""
from flask import Flask
from mimic.text_generator import TextGenerator
from mimic.text_generator_factory import TextGeneratorFactory
from flask import request
import json
app = Flask(__name__)


@app.route('/status', methods=['GET'])
def status():
    """status endpoint."""
    return "it's up!"


@app.route('/test-model', methods=['GET'])
def get_text():
    """Use pretrained model to generate text by the requested parameters."""
    data = json.loads(request.data)
    string_length = data["string_length"]
    seed_text = data["seed_text"]
    return "deferring this until we have pretrained models!!!"
