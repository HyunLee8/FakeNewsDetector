from utils import load_data
from preprocess import preprocessing_filter, one_hot_encoded
from model import build_model, best_threshold_value
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import nltk

class RunModel:
    def __init__(self, vocab_size=10000, max_length=100, embedding_dim=40):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.results = None
        self.history = None

    def get_model(self):
        return self.model

    def run_model(self): 
        nltk.download('stopwords')
        df = load_data()

        df['title'] = df['title'].apply(lambda x: preprocessing_filter(x, stem=True))
        df['one_hot_title'] = df['title'].apply(lambda x: one_hot_encoded(x, 5000))

        max_length = max(df['one_hot_title'].apply(len))
        padded_sequences = pad_sequences(df['one_hot_title'], maxlen=max_length, padding='pre')
        X = padded_sequences
        y = df['status'].values
        X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = build_model(vocab_size=5000, max_length=max_length)
        self.train_model(self.model, X_train, y_train, X_test, y_test)
        self.results = best_threshold_value(self.model, [0.4, 0.5, 0.6, 0.7, 0.8], X_test, y_test)

    def train_model(self, model, X_train, y_train, X_test, y_test):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        model_checkpoint = ModelCheckpoint('best_model.h5', mode='max', save_best_only=True, verbose=1)
        self.model = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=15,
            batch_size=64,
            callbacks=[early_stopping, model_checkpoint]
        )