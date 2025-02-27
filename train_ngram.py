import nltk
from nltk.util import ngrams
from collections import defaultdict

# Marathi Corpus for training
corpus = ["नमस्ते", "कसे आहात", "मी ठीक आहे", "शुभ प्रभात", "ज्ञान आहे शक्ती", "विद्या धन आहे"]

# Generate N-grams
n_gram_dict = defaultdict(int)

for sentence in corpus:
    words = list(sentence)
    for gram in ngrams(words, 3):  # Trigram Model
        n_gram_dict[gram] += 1

# Function to predict missing letters
def predict_missing(text):
    words = list(text.replace("...", " "))
    predictions = []
    
    for gram in ngrams(words, 3):
        if gram in n_gram_dict:
            predictions.append(max(n_gram_dict, key=n_gram_dict.get)[-1])  # Predict next character

    return "".join(predictions)

# Test the model
print("Predicted Text:", predict_missing("नम...ते"))
