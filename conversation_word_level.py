#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import re
import pickle
from keras.models import Model
from keras.models import load_model
from keras.layers import Input, Activation, Embedding, LSTM, concatenate, dot, Dense
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


max_seq_len = 30


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
        self.tokenizer, self.idx2word = self.load_tokenizer(tokenizer)
        self.model = load_model(model)
        self.encoder_model, self.decoder_model, self.attention_model = self.create_models()

    def load_tokenizer(self, file_name):
        with open(file_name, 'rb') as handle:
            tokenizer = pickle.load(handle)
            idx2word = dict(map(reversed, tokenizer.word_index.items()))
            return tokenizer, idx2word

    def create_models(self):
        hid_dim = 256

        # encoder
        encoder_inputs = self.model.input[0]
        encoded_seq, *encoder_states = self.model.layers[4].output
        encoder_model = Model(encoder_inputs, [encoded_seq] + encoder_states)

        # decoder
        decoder_states_inputs = [
            Input(shape=(hid_dim,)), Input(shape=(hid_dim,))]
        decoder_inputs = Input(shape=(1,))
        decoder_embedding = self.model.layers[3]
        decoder_embedded = decoder_embedding(decoder_inputs)
        decoder_lstm = self.model.layers[5]
        decoded_seq, * \
            decoder_states = decoder_lstm(
                decoder_embedded, initial_state=decoder_states_inputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs, [decoded_seq] + decoder_states)

        # attention
        encoded_seq_in, decoded_seq_in = Input(
            shape=(max_seq_len, hid_dim)), Input(shape=(1, hid_dim))
        score_dense = self.model.layers[6]
        score = score_dense(decoded_seq_in)
        score = dot([score, encoded_seq_in], axes=(2, 2))
        attention = Activation('softmax')(score)
        context = dot([attention, encoded_seq_in], axes=(2, 1))
        concat = concatenate([context, decoded_seq_in], axis=2)
        attention_dense = self.model.layers[11]
        attentional = attention_dense(concat)
        output_dense = self.model.layers[12]
        attention_outputs = output_dense(attentional)
        attention_model = Model([encoded_seq_in, decoded_seq_in], [
                                attention_outputs, attention])

        return encoder_model, decoder_model, attention_model

    def decode_sequence(self, input_seq, bos_eos, max_output_length=1000):
        encoded_seq, *states_value = self.encoder_model.predict(input_seq)

        target_seq = np.array(bos_eos[0])
        output_seq = bos_eos[0][:]
        attention_seq = np.empty((0, len(input_seq[0])))

        while True:
            decoded_seq, *states_value = self.decoder_model.predict([target_seq] + states_value)
            output_tokens, attention = self.attention_model.predict(
                [encoded_seq, decoded_seq])
            sampled_token_index = [np.argmax(output_tokens[0, -1, :])]
            output_seq += sampled_token_index
            attention_seq = np.append(attention_seq, attention[0], axis=0)

            if (sampled_token_index == bos_eos[1] or len(output_seq) > max_output_length):
                break

            target_seq = np.array(sampled_token_index)

        return output_seq, attention_seq

    def reply(self, input_seq):
        formated_input_seq = '<s> {} </s>'.format(clean_text(input_seq))
        tokenized_input_seq = pad_sequences(self.tokenizer.texts_to_sequences(
            [formated_input_seq]), padding='post', maxlen=max_seq_len)
        bos_eos = self.tokenizer.texts_to_sequences(['<s>', '</s>'])
        tokenized_output_seq, _ = self.decode_sequence(tokenized_input_seq, bos_eos)
        output_seq = ' '.join([self.idx2word[i]
                               for i in tokenized_output_seq if i not in [0, 1, 2, 3]])
        response = {
            "input_seq": input_seq,
            "output_seq": output_seq,
        }
        return response
