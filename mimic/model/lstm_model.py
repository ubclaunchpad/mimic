from mimic.model.model import Model

# Keras & Tensorflow imports
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import tensorflow as tf

# Useful util imports
import numpy as np
import string, os 

class LSTMModel(Model):

    def __init__(self):
        self.tokenizer = Tokenizer()

    def learn(self, text):
        # Create sequences to train the model - each is a sequence of tokens that represent characters
        inp_sequences, total_words = self.__get_sequence_of_tokens(text)
        predictors, label, max_sequence_len = self.__generate_padded_sequences(inp_sequences, total_words)
        self.max_sequence_len = max_sequence_len
        self.model = self.__create_model(max_sequence_len, total_words)
        self.model.summary()
        self.model.fit(predictors, label, epochs=100, verbose=5)

    def predict(self):
        # Arbitrary constants for now
        return self.__generate_text("Where art thou", 50, self.model, self.max_sequence_len)

    # TODO seq_len and batch_size are currently arbitrary
    def __get_sequence_of_tokens(self, text, seq_len=100, batch_size=200):

        # Clean text of weird characters
        text = "".join(v for v in text if v not in string.punctuation).lower()
        clean_txt = text.encode("utf8").decode("ascii",'ignore')

        # TODO: We need some method of splitting up the input text into chunks of a certain size
        # should be randomized so that we have a good diversity of text
        corpus = list((clean_txt[0+i:seq_len+i] for i in range(0, len(clean_txt), seq_len)))[0:batch_size]

        ## Tokenization of corpus
        self.tokenizer.fit_on_texts(corpus)
        total_words = len(self.tokenizer.word_index) + 1
        input_sequences = []
        for line in corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

        return input_sequences, total_words

    # This creates sequences such that all the sequences are the same length
    def __generate_padded_sequences(self, input_sequences, total_words):
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        
        predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
        label = ku.to_categorical(label, num_classes=total_words)
        return predictors, label, max_sequence_len

    # Creates the LSTM model to train
    # TODO Seems like there are lots of arbitrary constants here, experiment to get better results
    def __create_model(self, max_sequence_len, total_words):
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
        
        return model

    # Generate output text when given a trained model
    def __generate_text(self, seed_text, next_words, model, max_sequence_len):
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predicted = model.predict_classes(token_list, verbose=0)
            
            output_word = ""
            for word,index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " "+output_word
        return seed_text


