#!/usr/bin/env python

####################################################################
##                 Import Required Libraries Here                ##
####################################################################
import multiprocessing
import textwrap
import tensorflow as tf
import sys
import os
import argparse
import time
import re
import random
import gensim
import warnings
import nltk
import numpy as np
import pandas as pd
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda, Input
from keras.utils import np_utils
from keras.preprocessing import sequence
from keraspreprocessing.text import Tokenizer
from textblob import TextBlob, Word
from keras_preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Start Experiment Start Time
    start_time = time.time()

    # Instanciate the Parser and parse arguments
    parser = argparse.ArgumentParser(description="train mnist", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ####################################################################
    ##              Added Your HyperParameters here                   ##
    ####################################################################
    parser.add_argument('--epoch_count', type=int, default=5, help='the number of epochs')
    parser.add_argument('--drop_count', type=int, default=100, help='minimum number of occurances')
    
    args = parser.parse_args()

    # Print Hyperparameters used by AutoML experiment
    epoch_count = args.epoch_count
    print(">>> args.epoch_count received by trial = ", epoch_count)
    drop_count = args.drop_count
    print(">>> drop_count received by trial = ", drop_count)

    ####################################################################
    ##               Your Experiment starts here                      ##
    ####################################################################

    TRACE = False
    embedding_dim = 100

    def set_seeds_and_trace():
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(42)
        tf.random.set_seed(42)
        random.seed(42)
        if TRACE:
            tf.debugging.set_log_device_placement(True)

    def set_session_with_gpus_and_cores():
        cores = multiprocessing.cpu_count()
        gpus = len(tf.config.list_physical_devices('GPU'))
        config = tf.compat.v1.ConfigProto( device_count = {'GPU': gpus  , 'CPU': cores} , intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(config=config) 
        K.set_session(sess)

    set_seeds_and_trace()
    set_session_with_gpus_and_cores()
    warnings.filterwarnings('ignore')
    nltk.download('punkt')
    textblob_tokenizer = lambda x: TextBlob(x).words

    def preprocess_text(text, should_join=True):
        text = ' '.join(str(word) for word in textblob_tokenizer(textwrap))
        text = re.sub(r"([.,!?])", r" \1 ", text)
        text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
        if should_join:
            return ' '.join(gensim.utils.simple_preprocess(text))
        else:
            return gensim.utils.simple_preprocess(text)
    
    path = './call_center_intents.csv'
    intents = pd.read_csv(path, header=None, names=["intent", "query"])

    intents.intent.value_counts()
    intents_filtered = intents.groupby('intent').filter(lambda x: len(x) >= drop_count).reset_index()

    X = intents_filtered['query']
    y = intents_filtered.intent

    #Creating the corpus and tokenizing
    corpus_with_ix = [(ix, preprocess_text(sentence, should_join = True)) for ix, sentence in X.items() if type(sentence) == str and len(textblob_tokenizer(sentence)) > 3]
    corpus_df = pd.DataFrame(corpus_with_ix, columns=['index', 'text'])
    corpus_df
    y_filtered = y[corpus_df['index']]
    corpus = [preprocess_text(sentence, should_join=False) for ix, sentence in corpus_with_ix]

    def get_maximum_review_length(tokenized_corpus):
        maximum = 0
        for sentence in tokenized_corpus:
            candidate = len(sentence)
            if candidate > maximum:
                maximum = candidate
            return maximum
        
    max_review_length = get_maximum_review_length(corpus)
        

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    tokenized_corpus = tokenizer.texts_to_sequences(corpus)
    nb_samples = sum(len(s) for s in corpus)
    vocab_size = len(tokenizer.word_index) + 1
    final_X = np.zeros((len(tokenized_corpus), max_review_length))
    for ix, tokenized_sentence in enumerate(tokenized_corpus):
        tokenized_sentence.extend([0]*(max_review_length-len(tokenized_sentence)))
        final_X[ix] = tokenized_sentence

    y_factorized, intent_categories = pd.factorize(y_filtered)   
    y_factorized, intent_categories 

    path_to_glove_file = "./glove.6B.100d.txt"
    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    embedding_dim = 100
    num_tokens = vocab_size + 1
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))
    
    #Doing the train_test split and defining Tensorflow model

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(final_X, y_factorized, test_size = 0.3, random_state=42)
    X_train_tensor = tf.constant(X_train)
    X_test_tensor = tf.constant(X_test)
    y_train_tensor = tf.one_hot(tf.constant(y_train), len(intent_categories))
    y_test_tensor = tf.one_hot(tf.constant(y_test), len(intent_categories))
    X_train_tensor.shape, X_test_tensor.shape, y_train_tensor.shape, y_test_tensor.shape

    vocab_size

    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=max_review_length, embeddings_initializer=Constant(embedding_matrix), trainable=False))
    model.add(Dense(100, activation=leaky_relu))
    model.add(Dense(50, activation=leaky_relu))
    model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dim,)))
    model.add(Dense(50, activation=leaky_relu))
    model.add(Dense(len(intent_categories), activation='softmax'))
    model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(), metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    model.summary()

    # Clear any logs from previous runs
    #!rm -rf ./logs/

    #Train and log the model
    history = model.fit(X_train_tensor, y_train_tensor, epochs = epoch_count, validation_split=0.2, workers = 5, callbacks=[])  



    ####################################################################
    ##        Calculate and Print Out Your Metrics to STDOUT          ##
    ####################################################################

    # Score model accuracy and log it to stdout
    accuracy = history.history['accuracy'][-1]

    # Calculate Experiment Run Time in seconds
    end_time = time.time()
    runtime = round((end_time - start_time),2)

    print("Accuracy= ", accuracy)
    print("RunTime= ", runtime)



