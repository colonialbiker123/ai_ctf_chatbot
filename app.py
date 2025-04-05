from flask import Flask, request, jsonify, render_template
import random
import pickle
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

app = Flask(__name__)

# Load data
lemmatizer = WordNetLemmatizer()
intents = json.load(open('intents.json'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')


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


def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure how to respond to that."
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "Sorry, something went wrong."


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["GET"])
def get_bot_response():
    userText = request.args.get('msg')
    ints = predict_class(userText, model)
    res = get_response(ints, intents)
    return res


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
