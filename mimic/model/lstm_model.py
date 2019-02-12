from mimic.model.model import Model

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import tensorflow as tf

import numpy as np
import string, os 

class LSTMModel(Model):

    def __init__(self):
        self.tokenizer = Tokenizer()

    ''' Create sequences to train the model - each is a sequence of tokens that represent characters '''
    def learn(self, text):

        # TODO These are currently arbitrary - we can look at varying these depending on the size of the input text
        SEQ_LEN = 100
        BATCH_SIZE = 200

        # Clean text of weird characters
        text = "".join(v for v in text if v not in string.punctuation).lower()
        clean_txt = text.encode("utf8").decode("ascii",'ignore')

        # TODO: We need some method of splitting up the input text into chunks of a certain size
        # This should be considered along with creating SEQ_LEN & BATCH_SIZE parameters
        corpus = list((clean_txt[0+i:SEQ_LEN+i] for i in range(0, len(clean_txt), SEQ_LEN)))[0:BATCH_SIZE]

        ## Tokenization of corpus
        self.tokenizer.fit_on_texts(corpus)
        total_words = len(self.tokenizer.word_index) + 1
        input_sequences = []
        for line in corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

        # This creates sequences such that all the sequences are the same length
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        predictors = input_sequences[:,:-1]
        label = ku.to_categorical(input_sequences[:,-1], num_classes=total_words)

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

    def predict(self):
        # TODO this uses arbitrary constants for now
        seed_text = "where art thou"
        for _ in range(50):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre')
            predicted = self.model.predict_classes(token_list, verbose=0)
            
            output_word = ""
            for word,index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " "+output_word
        return seed_text
