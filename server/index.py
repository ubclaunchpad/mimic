"""Server."""
from flask import Flask
from mimic.text_generator import TextGenerator
from mimic.text_generator_factory import TextGeneratorFactory
from flask import request
import json
from mimic.pretrained_model_factory import PretrainedModelFactory
app = Flask(__name__)

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint."""
    return "it's up!"

@app.route('/lstm_trump_tweets_model', methods=['GET'])
def get_lstm_trump_text():
    """Use the LSTM trump tweets pretrained model to generate text by the requested parameters."""
    data = json.loads(request.data)
    string_length = data["string_length"]
    seed_text = data["seed_text"]
    pretrained_factory = PretrainedModelFactory()
    LSTM_trump_model = pretrained_factory.create_pretrained_LSTM_trump_tweets_generator()
    gen_text = LSTM_trump_model.generate_text(seed_text=seed_text, pred_len=int(string_length))
    return gen_text

