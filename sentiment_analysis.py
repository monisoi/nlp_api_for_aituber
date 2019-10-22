#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
# Utility
import time
import pickle

# KERAS
SEQUENCE_LENGTH = 300
# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)


class SentimentAnalysis():
    def __init__(self, tokenizer, model):
        with open(tokenizer, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.model = load_model(model)
        # model._make_predict_function()

    def decode_sentiment(self, score, include_neutral=True):
        if include_neutral:
            label = NEUTRAL
            if score <= SENTIMENT_THRESHOLDS[0]:
                label = NEGATIVE
            elif score >= SENTIMENT_THRESHOLDS[1]:
                label = POSITIVE

            return label
        else:
            return NEGATIVE if score < 0.5 else POSITIVE

    def predict(self, text, include_neutral=True):
        start_at = time.time()
        # Tokenize text
        x_test = pad_sequences(self.tokenizer.texts_to_sequences(
            [text]), maxlen=SEQUENCE_LENGTH)
        # Predict
        score = self.model.predict([x_test])[0]
        # Decode sentiment
        label = self.decode_sentiment(score, include_neutral=include_neutral)

        return {"label": label, "score": float(score),
                "elapsed_time": time.time() - start_at}
