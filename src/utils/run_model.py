from utils import load_data
from preprocess import preprocessing_filter
from model import build_model, best_threshold_value
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import nltk
import time

class RunModel:
    def __init__(self, vocab_size=5000, max_length=100, embedding_dim=40):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.tokenizer = None
        self.results = None
        self.history = None

    def get_model(self):
        return self.model

    def run_model(self): 
        nltk.download('stopwords')
        
        print("Loading data...")
        start = time.time()
        df = load_data()
        df = df.head(5000)
        print(f"Loaded {len(df)} rows in {time.time()-start:.2f}s")

        print("Preprocessing text...")
        start = time.time()
        df['title'] = df['title'].apply(lambda x: preprocessing_filter(x, stem=True))
        print(f"Preprocessing took {time.time()-start:.2f}s")

        print("Tokenizing...")
        start = time.time()
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(df['title'])
        sequences = self.tokenizer.texts_to_sequences(df['title'])
        print(f"Tokenizing took {time.time()-start:.2f}s")

        print("Padding...")
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding='pre')
        
        X = padded_sequences
        y = df['status'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples")
        print(f"Input shape: {X_train.shape}")
    
        self.model = build_model(vocab_size=self.vocab_size, max_length=self.max_length)
        self.train_model(X_train, y_train, X_test, y_test)
        self.results = best_threshold_value(self.model, [0.4, 0.5, 0.6, 0.7, 0.8], X_test, y_test)
    
        import pickle
        with open('tokenizer.pkl', 'wb') as f:
            pickle.dump(self.tokenizer, f)

    def train_model(self, X_train, y_train, X_test, y_test):
        import tensorflow as tf
        print("GPU Available:", tf.config.list_physical_devices('GPU'))
        print("TensorFlow version:", tf.__version__)
        
        # Callback to print every 10 batches
        from tensorflow.keras.callbacks import LambdaCallback
        batch_callback = LambdaCallback(
            on_batch_begin=lambda batch, logs: print(f"Starting batch {batch}...") if batch < 5 else None,
            on_batch_end=lambda batch, logs: print(f"Batch {batch}/561 - loss: {logs['loss']:.4f}") if batch < 5 or batch % 50 == 0 else None
        )
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', 
                                        mode='max', save_best_only=True, verbose=1)
        
        print("Starting training...")
        print("Compiling first batch (this can take 30-60 seconds)...")
        start = time.time()
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=15,
            batch_size=64,
            callbacks=[early_stopping, model_checkpoint, batch_callback],
            verbose=2  # Changed to 2 for simpler output
        )
        print(f"\nTraining completed in {time.time()-start:.2f}s")