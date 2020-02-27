#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import time
import pickle

SEQUENCE_LENGTH = 300
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)


class SentimentAnalysis():
    def __init__(self, tokenizer, model):
        with open(tokenizer, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.model = load_model(model)

    def decode_sentiment(self, score):
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE
        return label

    def predict(self, text):
        start_at = time.time()
        x_test = pad_sequences(self.tokenizer.texts_to_sequences(
            [text]), maxlen=SEQUENCE_LENGTH)
        score = self.model.predict([x_test])[0]
        label = self.decode_sentiment(score)
        return {"label": label, "score": float(score),
                "elapsed_time": time.time() - start_at}
