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

def train_model(model, X_train, y_train, X_test, y_test):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model_checkpoint = ModelCheckpoint('best_model.h5', mode='max', save_best_only=True, verbose=1)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        batch_size=64,
        callbacks=[early_stopping, model_checkpoint]
    )
    return history

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
