"""Translation model class."""
from mimic.model.model import Model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from numpy import argmax
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


class TranslationModel(Model):
    """An encoder-decoder LSTM model for machine translation."""

    def __init__(self, n_units=256):
        """Initialize the model with the number of cells in each LSTM layer."""
        self.n_units = n_units

    def learn(self, bilingual_text):
        """Trains the model."""
        # prepare source language tokenizer
        src_tokenizer = self._create_tokenizer(bilingual_text[:, 0])
        src_vocab_size = len(src_tokenizer.word_index) + 1
        src_length = self._max_length(bilingual_text[:, 0])

        # prepare target language tokenizer
        tar_tokenizer = self._create_tokenizer(bilingual_text[:, 1])
        tar_vocab_size = len(tar_tokenizer.word_index) + 1
        tar_length = self._max_length(bilingual_text[:, 1])

        # prepare training data
        trainX = self._encode_sequences(src_tokenizer, src_length,
                                        bilingual_text[:, 0])
        trainY = self._encode_sequences(tar_tokenizer, tar_length,
                                        bilingual_text[:, 1])
        trainY = self._encode_output(trainY, tar_vocab_size)

        # define and train the model
        model = Sequential()
        model.add(Embedding(src_vocab_size, self.n_units,
                            input_length=src_length, mask_zero=True))
        model.add(LSTM(self.n_units))
        model.add(RepeatVector(tar_length))
        model.add(LSTM(self.n_units, return_sequences=True))
        model.add(TimeDistributed(
            Dense(tar_vocab_size, activation='softmax')))

        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.fit(trainX, trainY, epochs=30, batch_size=64)

        self.model = model
        self.tokenizer = tar_tokenizer

    def predict(self, source):
        """Generate translation of a single phrase."""
        prediction = self.model.predict(source, verbose=0)[0]
        integers = [argmax(vector) for vector in prediction]
        target = list()
        for i in integers:
            word = self._word_for_id(i, self.tokenizer)
            if word is None:
                break
            target.append(word)
        return ' '.join(target)

    def _create_tokenizer(self, lines):
        """Create tokenizer and fit on text."""
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    def _max_length(self, lines):
        """Get max sentence length."""
        return max(len(line.split()) for line in lines)

    def _encode_sequences(self, tokenizer, length, lines):
        """Encode and pad sequences."""
        # integer encode sequences
        X = tokenizer.texts_to_sequences(lines)
        # pad sequences with 0 values
        X = pad_sequences(X, maxlen=length, padding='post')
        return X

    def _encode_output(self, sequences, vocab_size):
        """One hot encode target sequence."""
        ylist = list()
        for sequence in sequences:
            encoded = to_categorical(sequence, num_classes=vocab_size)
            ylist.append(encoded)
        y = array(ylist)
        y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
        return y

    def _word_for_id(self, integer, tokenizer):
        """Map an interger to a word."""
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None
