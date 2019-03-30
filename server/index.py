"""Server."""
from flask import Flask
from mimic.text_generator import TextGenerator
from mimic.text_generator_factory import TextGeneratorFactory
from flask import request
import json

from mimic.pretrained_model_factory import PretrainedModelFactory
from flask_cors import CORS
app = Flask(__name__)

factory = PretrainedModelFactory()
lstm_trump = factory.create_pretrained_LSTM_trump_tweets_generator()
markov_trump = factory.create_pretrained_markov_chain_trump_tweets_generator()
lstm_sp = factory.create_pretrained_LSTM_shakespeare_text_generator()
markov_sp = factory.create_pretrained_markov_chain_shakespeare_text_generator()

CORS(app)

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint."""
    return "it's up!"


@app.route('/test-model', methods=['POST'])
def get_text():
    """Use pretrained model to generate text by the requested parameters."""
    data = json.loads(request.data)
    string_length = data["string_length"]
    seed_text = data["seed_text"]
    return json.dumps("deferring this until we have pretrained models!!!")


@app.route('/model/lstm/trump', methods=['POST'])
def get_lstm_trump_text():
    """Use the LSTM trump tweets model to generate text."""
    data = json.loads(request.data)
    sl = data["string_length"]
    st = data["seed_text"]
    gen_text = lstm_trump.generate_text(seed_text=st, pred_len=int(sl))
    return json.dumps(gen_text+"")


@app.route('/model/markov/trump', methods=['POST'])
def get_markov_trump_text():
    """Use the markov trump tweets model to generate text."""
    data = json.loads(request.data)
    sl = data["string_length"]
    st = data["seed_text"]
    gen_text = markov_trump.generate_text(seed_text=st, pred_len=int(sl))
    return json.dumps(gen_text+"")


@app.route('/model/lstm/shakespeare', methods=['POST'])
def get_lstm_shakespeare_text():
    """Use the LSTM shakespeare model to generate text."""
    data = json.loads(request.data)
    sl = data["string_length"]
    st = data["seed_text"]
    gen_text = lstm_sp.generate_text(seed_text=st, pred_len=int(sl))
    return json.dumps(gen_text+"")


@app.route('/model/markov/shakespeare', methods=['POST'])
def get_markov_shakespeare_text():
    """Use the markov chain shakespeare model to generate text."""
    data = json.loads(request.data)
    sl = data["string_length"]
    st = data["seed_text"]
    gen_text = markov_sp.generate_text(seed_text=st, pred_len=int(sl))
    return json.dumps(gen_text+"")