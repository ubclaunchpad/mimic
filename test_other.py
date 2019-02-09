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

# Test document
SHAKESPEARE_TXT = 'content/shakespeare.txt'

tokenizer = Tokenizer()

# TODO: We need some method of splitting up the input text into chunks of a certain size
# should be randomized so that we have a good diversity of text
def get_sequences_from_text(seq_len=100, batch_size=200):
    file = open(SHAKESPEARE_TXT, "r") 
    txt = file.read() 
    return list((txt[0+i:seq_len+i] for i in range(0, len(txt), seq_len)))[0:batch_size]

# Strips weird characters from text
def clean_text(txt):
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt 


def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

# This creates sequences such that all the sequences are the same length
def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

# Creates the LSTM model to train
def create_model(max_sequence_len, total_words):
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
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()

# Create the corpus
corpus = [clean_text(x) for x in get_sequences_from_text()]
# Not sure why the below is there
# corpus[:10]

inp_sequences, total_words = get_sequence_of_tokens(corpus)
# Not sure why the below is there
# inp_sequences[:10]

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)

model = create_model(max_sequence_len, total_words)
model.summary()
model.fit(predictors, label, epochs=100, verbose=5)

print(generate_text("Where art thou", 50, model, max_sequence_len))
