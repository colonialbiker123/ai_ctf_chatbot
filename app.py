from flask import Flask, request, jsonify
import random
import numpy as np
import nltk
import pickle
import json
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load model and data
model = tf.keras.models.load_model('chatbot_model.h5')
intents = json.load(open('intents.json'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list):
    if not intents_list:
        return "I'm not sure how to respond to that."
    tag = intents_list[0]['intent']
    for i in intents['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure how to respond to that."

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    intents_list = predict_class(user_message)
    response = get_response(intents_list)
    return jsonify({"response": response})

@app.route("/", methods=["GET"])
def home():
    return "CTF Chatbot is live!"

if __name__ == "__main__":
    app.run(debug=True)
