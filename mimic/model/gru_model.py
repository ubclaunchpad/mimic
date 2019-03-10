"""GRU model class."""

from model import Model
# import mimic.util as utils
import time
import os
import sys

import numpy as np
import random
import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


class GRUModel(Model):
    """Model for text generation using GRU RNN machine learning."""
    def __init__(self):
        self.model = None
        self.vocab = None
        self.char2idx = None
        self.idx2char = None
        self.text = None

    def learn(self, text):
        """Initialize a GRU RNN model and train it using a given text."""
        seq_length = 100        # Number of chars processed in a sequence
        embedding_dim = 256     # Embedding dimension
        rnn_units = 1024        # Number of rnn units
        batch_size = 64          # Batch size
        epochs = 1              # Number of training cycles (epochs)
        buffer_size = 10000     # Buffer used in shuffling training order

        # text = utils.clean_text(text)
        self.text = text
        self.vocab = sorted(set(text))
        self.char2idx = {u: i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
        text_as_int = np.array([self.char2idx[c] for c in text])
        examples_per_epoch = len(text) // seq_length

        # Create training examples / targets
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
        dataset = sequences.map(self.split_input_target)

        steps_per_epoch = examples_per_epoch // batch_size
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

        # Length of vocabulary in chars
        vocab_size = len(self.vocab)

        # Build model outline
        self.model = self.build_model(vocab_size=vocab_size,
                                      embedding_dim=embedding_dim,
                                      rnn_units=rnn_units,
                                      batch_size=batch_size)

        # Attach optimizer and loss function
        self.model.compile(
            optimizer=tf.train.AdamOptimizer(),
            loss=self.loss
        )

        # Directory where the checkpoints will be saved
        checkpoint_dir = "./training_checkpoints"
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "cpkt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True
        )

        # Actual training and fitting of model
        history = self.model.fit(dataset.repeat(), epochs=epochs, steps_per_epoch=steps_per_epoch,
                                 callbacks=[checkpoint_callback])

    def load(self, text):
        embedding_dim = 256  # Embedding dimension
        rnn_units = 1024  # Number of rnn units
        batch_size = 1  # Batch size
        checkpoint_dir = "/Users/irvinodjuana/PycharmProjects/mimic/mimic/model/training_checkpoints"

        self.text = text
        self.vocab = sorted(set(text))
        self.char2idx = {u: i for i, u in enumerate(self.vocab)}
        self.idx2char = np.array(self.vocab)
        vocab_size = len(self.vocab)

        self.model = self.build_model(vocab_size, embedding_dim, rnn_units, batch_size=batch_size)
        self.model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        self.model.build(tf.TensorShape([1, None]))
        # self.model.summary()

    def predict(self):
        """Generate text using a learned GRU model"""
        num_generate = 3000         # Number of characters to generate
        text_generated = []         # Empty string to store our results
        temperature = 0.7         # temperature : low = more predictable text, high = more surprising text

        all_words = self.text.split()
        seed_word = all_words[random.randint(0, len(all_words) - 1)]

        # Converting our start string to numbers (vectorizing)
        input_eval = [self.char2idx[s] for s in seed_word]
        input_eval = tf.expand_dims(input_eval, 0)

        # Here batch size = 1
        self.model.reset_states()
        for i in range(num_generate):
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0)

            # Using a multinomial distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1, 0].numpy()

            # We pass the predicted word as the next input to the model
            # along with the previous hidden state

            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.idx2char[predicted_id])

        return seed_word + ''.join(text_generated)

    @staticmethod
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    @staticmethod
    def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
        if tf.test.is_gpu_available():
            rnn = tf.keras.layers.CuDNNGRU
        else:
            import functools
            rnn = functools.partial(GRU, recurrent_activation='sigmoid')

        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]))
        model.add(rnn(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True))
        model.add(Dense(vocab_size))

        return model

    @staticmethod
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def main():
    path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    gru_model = GRUModel()
    # gru_model.learn(text)
    gru_model.load(text)
    print(gru_model.predict())


if __name__ == "__main__":
    main()
