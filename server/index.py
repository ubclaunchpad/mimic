"""Server."""
from flask import Flask
from mimic.text_generator import TextGenerator
from mimic.text_generator_factory import TextGeneratorFactory
from flask import request
import json
app = Flask(__name__)


@app.route('/test-endpoint', methods=['GET'])
def testEndpoint():
    """Test endpoint."""
    return "it's up!"


@app.route('/test-model', methods=['GET'])
def getText():
    """Use pretrained model to generate text by the requested parameters."""
    data = json.loads(request.data)
    stringLength = data["stringLength"]
    seedText = data["seedText"]
    return "deferring this until we have pretrained models!!!"
