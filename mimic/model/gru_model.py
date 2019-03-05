"""GRU model class."""

from mimic.model import Model
import time
import os
import sys

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

import tf.keras.Sequential as sequential
import tf.keras.Embedding as embedding
import tf.keras.layers.GRU as gru
import tf.keras.layers.Dense as dense

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


class GRUModel(Model):
    """Model for text generation using GRU RNN machine learning"""
    def __init__(self):
        pass



def main():
    # print(tf.reduce_sum(tf.random_normal([1000, 1000])))
    path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                           'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])
    #
    # print('{')
    # for char, _ in zip(char2idx, range(20)):
    #     print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
    # print('  ...\n}')
    #
    # print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

    # Max sequence length of chars we want as single input
    seq_length = 10
    examples_per_epoch = len(text) // seq_length

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    #
    # for i in char_dataset.take(5):
    #     print(idx2char[i.numpy()])

    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
    #
    # for item in sequences.take(5):
    #     print(repr(''.join(idx2char[item.numpy()])))

    dataset = sequences.map(split_input_target)
    #
    # for input_example, target_example in dataset.take(1):
    #     print("Input:  " + repr(''.join(idx2char[input_example.numpy()])))  # Print char repr of each input batch
    #     print("Target: " + repr(''.join(idx2char[target_example.numpy()])))
    #
    #     for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    #         print("Step {:4d}".format(i))
    #         print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    #         print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

    # Batch size
    batch_size = 64
    steps_per_epoch = examples_per_epoch // batch_size

    # Buffer size to shuffle to sequence
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    buffer_size = 10000

    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    print(dataset)

    print("Text Length: " + str(len(text_as_int)))
    print("Examples per epoch: " + repr(examples_per_epoch))
    print("Steps per epoch: " + repr(steps_per_epoch))


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


if __name__ == "__main__":
    main()
