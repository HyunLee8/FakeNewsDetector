from preprocess import preprocessing_filter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import RunModel
import pickle

def main():
    runner = RunModel()
    runner.run_model()
    model = runner.get_model()
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    text = input("Enter news title to check if it's fake: ")
    preprocessed_text = preprocessing_filter(text, stem=True)
    encoded_text = tokenizer.texts_to_sequences([preprocessed_text])[0]
    padded_encoded_text = pad_sequences([encoded_text], maxlen=100, padding='pre')

    prediction = model.predict(padded_encoded_text)
    prediction_label = 'No, It is not fake' if prediction[0][0] < 0.4 else 'Yes, this News is fake'
    print(prediction_label)

if __name__ == "__main__":
    main()