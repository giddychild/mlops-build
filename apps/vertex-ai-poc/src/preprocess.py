import multiprocessing
import tensorflow as tf
import sys
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda, Input
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from textblob import TextBlob, Word
from keras_preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.nn import leaky_relu
from keras_preprocessing.text import tokenizer_from_json
import numpy as np
import re
import random
import os
import pandas as pd
import gensim
import warnings
import nltk
import io
import json
nltk.download('punkt')

embedding_dim = 100
vocab_size = 698
num_tokens = vocab_size + 1
max_review_length = 43
textblob_tokenizer = None
tokenizer = None
model = None

# Define the intent categories
level_intents = ['flight', 'flight_time', 'airfare', 'aircraft','ground_service', 'airport', 'airline', 'distance','abbreviation', 'ground_fare', 'quantity', 'city', 'capacity', 'flight#airfare']

# Defining prepocessing text function
def preprocess_text(text, should_join=True):
  text = ' '.join(str(word) for word in textblob_tokenizer(text))
  text = re.sub(r"([.,!?])", r" \1 ", text)
  text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
  if should_join:
    return ' '.join(gensim.utils.simple_preprocess(text))
  else:
    return gensim.utils.simple_preprocess(text)

def load_model_tokenizer():
  # Loading the tokenizer
  with open('tokenizer_v1.0.json') as f:
    data = json.load(f)
    global tokenizer
    tokenizer = tokenizer_from_json(data)

  global textblob_tokenizer
  textblob_tokenizer = lambda x: TextBlob(x).words

  # Prepare embedding matrix
  embedding_matrix = np.zeros((num_tokens, embedding_dim))

  # Define the Tensorflow model layers
  global model
  model = Sequential()
  model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], input_length=max_review_length, embeddings_initializer=Constant(embedding_matrix), trainable=False))
  model.add(Dense(100, activation=leaky_relu))
  model.add(Dense(50, activation=leaky_relu))
  model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dim,)))
  model.add(Dense(50, activation=leaky_relu))
  model.add(Dense(len(level_intents), activation='softmax'))

  # Loading model weights
  model.load_weights('call_center_intent_model_v1.0.h5')

# Defining the predict function
def predict_question(question):
  #Tokenize the question
  tokenized_question = np.zeros((1, max_review_length))
  for ix, tokenized_sentence in enumerate(tokenizer.texts_to_sequences([preprocess_text(sentence, should_join=True) for sentence in question])):
    tokenized_sentence.extend([0]*(max_review_length-len(tokenized_sentence)))
    tokenized_question[ix] = tokenized_sentence

  #Make the prediction
  prediction = model.predict(tokenized_question)
  prediction

  #Find highest intent with highest predicted value and display that category back as text
  result = np.asarray(tf.math.argmax(prediction, axis=1), dtype=np.int32)
  predicted_intent = level_intents[result[0]]

  return predicted_intent

