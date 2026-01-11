from preprocess import preprocessing_filter, one_hot_encoded, word_embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import RunModel

runner = RunModel()
runner.run_model()
model = runner.get_model()

def main():
    # call model method will go here
    text = input("Enter news title to check if it's fake: ")
    preprocessed_text = preprocessing_filter(text)
    encoded_text = word_embedding(preprocessed_text)
    padded_encoded_text = pad_sequences([encoded_text], maxlen=100000, padding='pre')

    prediction = model.predict(padded_encoded_text)
    prediction_label = 'No, It is not fake' if prediction[0][0] < 0.4 else 'Yes, this News is fake'

if __name__ == "__main__":
    main() 