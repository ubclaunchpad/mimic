"""Server."""
from flask import Flask
from mimic.text_generator import TextGenerator
from mimic.text_generator_factory import TextGeneratorFactory
from flask import request
import json
from mimic.pretrained_model_factory import PretrainedModelFactory
app = Flask(__name__)
pretrained_factory = PretrainedModelFactory()


@app.route('/status', methods=['GET'])
def status():
    """Status endpoint."""
    return "it's up!"


@app.route('/lstm_trump_tweets_model', methods=['GET'])
def get_lstm_trump_text():
    """Use the LSTM trump tweets model to generate text."""
    data = json.loads(request.data)
    sl = data["string_length"]
    st = data["seed_text"]
    model = pretrained_factory.create_pretrained_LSTM_trump_tweets_generator()
    gen_text = model.generate_text(seed_text=st, pred_len=int(sl))
    return gen_text
