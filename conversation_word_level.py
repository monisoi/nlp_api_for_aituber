#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import re
import pickle
from keras.models import Model, load_model
from keras.layers import Input
from keras.preprocessing.sequence import pad_sequences


SEQUENCE_LENGTH = 30
MAX_OUTPUT_LENGTH = 100


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"'#/@;:<>{}`+=~|.!?,]", "", text)
    return text


class Conversation():
    def __init__(self, tokenizer, model):
        self.tokenizer, self.word2index, self.index2word = self.load_tokenizer(tokenizer)
        self.model = load_model(model)
        self.encoder_model, self.decoder_model = self.create_models()

    def load_tokenizer(self, file_name):
        with open(file_name, 'rb') as handle:
            tokenizer = pickle.load(handle)
            word2index = tokenizer.word_index
            index2word = dict(map(reversed, word2index.items()))
            return tokenizer, word2index, index2word

    def create_models(self):
        HIDDEN_DIM = 256

        # encoder
        encoder_inputs = self.model.input[0]
        _, *encoder_states = self.model.layers[4].output
        encoder_model = Model(encoder_inputs, encoder_states)

        # decoder
        decoder_inputs = Input(shape=(1,))
        decoder_embedding_layer = self.model.layers[3]
        decoder_embedded = decoder_embedding_layer(decoder_inputs)
        decoder_lstm_layer = self.model.layers[5]
        decoder_states_inputs = [Input(shape=(HIDDEN_DIM,)), Input(shape=(HIDDEN_DIM,))]
        decoder_lstm, *decoder_states = decoder_lstm_layer(
            decoder_embedded,
            initial_state=decoder_states_inputs)
        decoder_dense_layer = self.model.layer[6]
        decoder_outputs = decoder_dense_layer(decoder_lstm)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return encoder_model, decoder_model

    def decode_sequence(self, input_seq):
        formated_input_seq = '<s> {} </s>'.format(clean_text(input_seq))
        tokenized_input_seq = pad_sequences(
            self.tokenizer.texts_to_sequences([formated_input_seq]),
            padding='post',
            maxlen=SEQUENCE_LENGTH)
        bos = [self.word2index['<s>']]
        eos = [self.word2index['</s>']]
        states = self.encoder_model.predict(tokenized_input_seq)
        target = np.array(bos)
        output_seq = bos
        for i in range(MAX_OUTPUT_LENGTH):
            tokens, *states = self.decoder_model.predict([target] + states)
            output_index = [np.argmax(tokens[0, -1, :])]
            output_seq += output_index
            if output_index == eos:
                break
            target = np.array(output_index)
        output_seq = ' '.join([self.index2word[i] for i in output_seq if i not in bos + eos])
        return output_seq

    def reply(self, input_seq):
        output_seq = self.decode_sequence(input_seq)
        response = {
            "input_seq": input_seq,
            "output_seq": output_seq,
        }
        return response
