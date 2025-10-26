import regex as re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tensorflow.keras.preprocessing.text import one_hot

stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

def preprocessing_filter(text, stem=False):
    text_cleaning = r"\b0\S*|\b[^A-Za-z0-9]+"
    text = re.sub(text_cleaning, ' ', text)
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                stemmer = SnowballStemmer('english')
                token = stemmer.stem(token)
            tokens.append(token)
    return " ".join(tokens)
      
def one_hot_encoded(text,vocab_size=5000, max_length = 40):
    hot_encoded = one_hot(text,vocab_size)
    return hot_encoded

def word_embedding(text):
	preprocessed_text = preprocessing_filter(text)
	return one_hot_encoded(preprocessed_text)