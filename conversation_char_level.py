#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from keras.layers import Dense, LSTM, Input
from keras.models import Model
from keras.models import load_model


latent_dim = 256
num_encoder_tokens = 37
num_decoder_tokens = 39
max_decoder_seq_length = 301

input_token_index = {' ': 0, '0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8,
                     '8': 9, '9': 10, 'a': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15, 'f': 16,
                     'g': 17, 'h': 18, 'i': 19, 'j': 20, 'k': 21, 'l': 22, 'm': 23, 'n': 24,
                     'o': 25, 'p': 26, 'q': 27, 'r': 28, 's': 29, 't': 30, 'u': 31, 'v': 32,
                     'w': 33, 'x': 34, 'y': 35, 'z': 36}
target_token_index = {'\t': 0, '\n': 1, ' ': 2, '0': 3, '1': 4, '2': 5, '3': 6, '4': 7, '5': 8,
                      '6': 9, '7': 10, '8': 11, '9': 12, 'a': 13, 'b': 14, 'c': 15, 'd': 16,
                      'e': 17, 'f': 18, 'g': 19, 'h': 20, 'i': 21, 'j': 22, 'k': 23, 'l': 24,
                      'm': 25, 'n': 26, 'o': 27, 'p': 28, 'q': 29, 'r': 30, 's': 31, 't': 32,
                      'u': 33, 'v': 34, 'w': 35, 'x': 36, 'y': 37, 'z': 38}
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


class Conversation():
    def __init__(self, model):
        self.model = load_model(model)
        self.encoder_model = self.create_encoder()
        self.decoder_model = self.create_decoder()

    def create_encoder(self):
        encoder_inputs = self.model.input[0]  # input_1
        encoder_outputs, state_h, state_c = self.model.layers[2].output
        encoder_states = [state_h, state_c]
        return Model(encoder_inputs, encoder_states)

    def create_decoder(self):
        decoder_inputs = self.model.input[1]  # input_2
        decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
        decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = self.model.layers[3]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_dense = self.model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        return Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    def decode_sequence(self, input_seq):
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, target_token_index['\t']] = 1.
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char
            if (sampled_char == '\n' or
                    len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
            states_value = [h, c]
        return decoded_sentence

    def reply(self, input_seq):
        onehot_input_seq = np.zeros(
            (1, len(input_seq), num_encoder_tokens), dtype='float32')
        for i, char in enumerate(input_seq):
            onehot_input_seq[0, i, input_token_index[char]] = 1.
        response = {
            "input_seq": input_seq,
            "output_seq": self.decode_sequence(onehot_input_seq),
        }
        return response
