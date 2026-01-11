from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

embedded_features = 40

def build_model(vocab_size, max_length, embedding_dim=embedded_features):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(100, return_sequences=True),
        Dropout(0.3),
        LSTM(100),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def best_threshold_value(model, thresholds:list, X_test, y_test):
    accuracies = []
    for thresh in thresholds:
        ypred = model.predict(X_test)
        ypred = (ypred > thresh).astype(int)
        accuracies.append(accuracy_score(y_test, ypred))
    return pd.DataFrame({
        'Threshold': thresholds,
        'Accuracy': accuracies
    })
