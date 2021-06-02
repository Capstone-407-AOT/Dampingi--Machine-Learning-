# Imports
from flask import Flask, render_template, request, jsonify
from flask_mysqldb import MySQL
import nltk
import datetime
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import pickle

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'MyDB'

mysql = MySQL(app)

stemmer = LancasterStemmer()
seat_count = 50

with open("training.json") as file:
    data = json.load(file)
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

# Function to process input


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)


tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

# Loading existing model from disk
model = tflearn.DNN(net)
model.load("model.tflearn")


app = Flask(__name__)


data_dibutuhkan = {
    "data_awal": {
        "nama": ""
    },
    "kategori": {
        "kekerasan_anak": {

        },
        "kekerasan_perempuan": {

        }
    }
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/mulai/:id_percakapan ')
def get_mulai_percakapan():

    global seat_count

    message = request.args.get('msg')

    if message:
        message = message.lower()
        results = model.predict([bag_of_words(message, words)])[0]
        result_index = np.argmax(results)
        tag = labels[result_index]

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']

                response = random.choice(responses)
        return str(response)
    return "Missing Data!"


@app.route('/percakapan/:id_percakapan ')
def percakapan():

    global seat_count

    message = request.args.get('msg')

    if message:
        message = message.lower()
        results = model.predict([bag_of_words(message, words)])[0]
        result_index = np.argmax(results)
        tag = labels[result_index]

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']

                response = random.choice(responses)
        return str(response)
    return "Missing Data!"


@app.route('/tutup_percakapan/:id_percakapan ')
def tutup_percakapan():

    global seat_count

    message = request.args.get('msg')

    if message:
        message = message.lower()
        results = model.predict([bag_of_words(message, words)])[0]
        result_index = np.argmax(results)
        tag = labels[result_index]

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']

                response = random.choice(responses)
        return str(response)
    return "Missing Data!"


if __name__ == "__main__":
    app.run()
