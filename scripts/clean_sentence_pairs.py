"""Script to clean sentence pairs in a text file."""
import argparse
import os
import string
import re
import logging
import gzip
from pickle import dump
from unicodedata import normalize
from numpy import array


def load_doc(filename):
    """Load the text file."""
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text


def to_pairs(doc):
    """Split a loaded document into pairs of sentences."""
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs


def clean_pairs(pairs):
    """Clean a list of pairs."""
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in pairs:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            # tokenize on white space
            line = line.split()
            # convert to lowercase
            line = [word.lower() for word in line]
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            # remove tokens with numbers in them
            line = [word for word in line if word.isalpha()]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)


def save_clean_data(sentences, filename):
    """Save a list of clean sentences to a gz file."""
    f = gzip.open(filename, 'wb')
    dump(sentences, f)
    f.close()
    logging.info('Saved: %s' % filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_filename",
                        help="tab-delimited sentence pairs text file")
    parser.add_argument("output_filename", help="output gz file")

    io_args = parser.parse_args()
    input_filename = io_args.input_filename
    output_filename = io_args.output_filename

    # load tab-delimited sentence pairs text file
    doc = load_doc(input_filename)
    # split into english-german pairs
    pairs = to_pairs(doc)
    # clean sentences
    clean_pairs = clean_pairs(pairs)
    # save clean pairs to file
    save_clean_data(clean_pairs, output_filename)
