"""LSTM model class."""
from mimic.model.model import Model
import mimic.util as utils

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import load_model
import keras.utils as ku
import tensorflow as tf

import numpy as np
import string
import os
from random import randint
import logging


class LSTMModel(Model):
    """ML Model for Text Prediction using the LSTM Model with Keras."""

    def __init__(self, sequenceLength, predictionLength):
        """Initialize the LSTM Model."""
        self.tokenizer = Tokenizer()
        self.seqLen = sequenceLength
        self.predLen = predictionLength
        logging.info('Initialized LSTM Model')

    def learn(self, text):
        """Use input text to train the LSTM model."""
        # Clean & verify text
        logging.info('Cleaning and verifying text')

        clean_txt = utils.clean_text(text)
        txt_len = len(clean_txt)
        utils.verify_text(clean_txt)
        self.cleaned_input_text = clean_txt
        logging.info('Tokenizing Corpus')
        corpus = list(clean_txt[0+i:self.seqLen+i] for i in range(0,
                                                                  txt_len,
                                                                  self.seqLen))
        # Tokenization of corpus
        self.tokenizer.fit_on_texts(corpus)
        total_words = len(self.tokenizer.word_index) + 1
        input_sequences = []
        for line in corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        # This makes sequences such that all the sequences are the same length
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences,
                                                 maxlen=max_sequence_len,
                                                 padding='pre'))
        predictors = input_sequences[:, :-1]
        label = ku.to_categorical(input_sequences[:, -1],
                                  num_classes=total_words)
        # Creates the LSTM model to train
        input_len = max_sequence_len - 1
        model = Sequential()
        # Add Input Embedding Layer
        model.add(Embedding(total_words, 10, input_length=input_len))
        # Add Hidden Layer 1 - LSTM Layer
        model.add(LSTM(100))
        model.add(Dropout(0.1))
        # Add Output Layer
        model.add(Dense(total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(predictors, label, epochs=100, verbose=5)

        self.max_sequence_len = max_sequence_len
        self.model = model
        logging.info('Tokenization successfully completed')

    def predict(self):
        """Generate a sequence of text based on prior training."""
        logging.info('Generating text')
        split_input_text = self.cleaned_input_text.split()
        # Picks a random word from the input text as seed
        seed_text = split_input_text[randint(0, len(split_input_text)-1)]

        # Numerical input here is the # of words to generate
        for _ in range(self.predLen):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list],
                                       maxlen=self.max_sequence_len-1,
                                       padding='pre')
            predicted = self.model.predict_classes(token_list, verbose=0)
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " "+output_word
        logging.info('Text successfully generated.')
        return seed_text

    def load_pretrained_model(self, filepath):
        """
        Load a pretrained LSTM model from a filepath.

        Loads trained LSTM and returns true if loaded
        or false if an error occured.
        """
        try:
            self.model = load_model(filepath)
            return True
        except (ImportError, ValueError) as e:
            logging.error(e)
            return False

    def save_trained_model(self, dir_path, filename):
        """Save a pretrained LSTM model to a file and return the file path."""
        filepath = os.path.join(dir_path, filename + ".h5")
        self.model.save(filepath)
        return filepath
