import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from text_preprocessing import preprocessing_filter, one_hot_encoded, word_embedding
from data_utils import load_data
from model_training import build_model, train_model, best_threshold_value

nltk.download('stopwords')

df = load_data('Fake.csv', 'True.csv')
print(f"Loaded {len(df)} total samples")

df['title'] = df['title'].apply(lambda x: preprocessing_filter(x, stem=True))
df['one_hot_title'] = df['title'].apply(lambda x: one_hot_encoded(x, 5000))

max_length = max(df['one_hot_title'].apply(len))
padded_sequences = pad_sequences(df['one_hot_title'], maxlen=max_length, padding='pre')

X = padded_sequences
y = df['status'].values

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = build_model(vocab_size=5000, max_length=max_length)
print(model.summary())
train_model(model, X_train, y_train, X_test, y_test)

results = best_threshold_value(model, [0.4, 0.5, 0.6, 0.7, 0.8], X_test, y_test)
print(results)

def prediction_input_processing(text):
    preprocessed_text = preprocessing_filter(text)
    encoded_text = word_embedding(preprocessed_text)
    padded_encoded_text = pad_sequences([encoded_text], maxlen=max_length, padding='pre')
    
    prediction = model.predict(padded_encoded_text)
    prediction_label = 'No, It is not fake' if prediction[0][0] < 0.4 else 'Yes, this News is fake'
	
    return prediction_label

while True:
    user_input = input("Enter news title to check if it's fake (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    result = prediction_input_processing(user_input)
    print(result) 