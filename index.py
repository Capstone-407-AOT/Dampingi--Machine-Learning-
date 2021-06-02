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
import uuid
  

# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = 'root'
# app.config['MYSQL_DB'] = 'dampingi_chatbot'

# mysql = MySQL(app)

nltk.download('punkt')

stemmer = LancasterStemmer()

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
        "nama": "",
        "nik":"",
        "alamat":"",
        "diagnosa":"",
        "kategori":""
    },
    "kategori": {
        "kekerasan_anak": {
            "jenis":"",
            "dimana":"",
            "kapan":"",
            "siapa":"",
            "yang_melapor":""
        },
        "kekerasan_perempuan": {
            "jenis":"",
            "dimana":"",
            "kapan":"",
            "siapa":"",
            "yang_melapor":""
        }
    }
}

pertanyaan = {
    "konfirmasi_nama":{
        "mengisi":"nama",
        "accept":["affirmative","negative"],
        "options":["benar","salah"],
        "pesan":""
    },
    "konfirmasi_nik":{
        "mengisi":"nik",
        "accept":["affirmative","negative"],
        "options":["benar","salah"]
    },
    "konfirmasi_alamat":{
        "mengisi":"alamat",
        "accept":["affirmative","negative"],
        "options":["benar","salah"]
    },
    "konfirmasi_nohp":{
        "mengisi":"nohp",
        "accept":["affirmative","negative"],
        "options":["benar","salah"]
    }
}
daftar_pertanyaan = ["nama","nik","alamat","nohp"]

formulir_data_awal = {

}

formulir = {}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/mulai')
def get_mulai_percakapan():

    # generate id percakapan

    id_percakapan = uuid.uuid4().hex

    # inisialisasi data mulai percakapan
    ## tentukan data yang dibutuhkan
    formulir[id_percakapan] = {}

    ## simpan ke db
    # cur = mysql.connection.cursor()
    # cur.execute("INSERT INTO percakapan(id_percakapan) VALUES (%s)",
    #             (id))
    # mysql.connection.commit()
    # cur.close()

    # terima dan cek data pengguna 
    nama = request.args.get('nama')
    nik = request.args.get('nik')
    nohp = request.args.get('nohp')
    alamat = request.args.get('alamat')
    print(nama, nik, alamat)

    if nama:
        formulir[id_percakapan]['nama']=nama
    if nik:
        formulir[id_percakapan]['nik']=nik
    if alamat:
        formulir[id_percakapan]['alamat']=alamat
    if nohp:
        formulir[id_percakapan]['nohp']=nohp
    print("formulir", formulir[id_percakapan])

    
    # mulai percakapan awal
    return jsonify(
        error=False,
        formulir=formulir[id_percakapan],
        id_percakapan=id_percakapan,
        message=["Halo!","Selamat pagi", "Saya merupakan agent otomatis dari Dampingi. Saya akan membantu anda hari ini!", "Sebelumnya saya ingin mengonfirmasi beberapa hal.", "Benarkah nama kamu %s?" % formulir[id_percakapan]['nama']],
        context="konfirmasi_nama",
        options=pertanyaan["konfirmasi_nama"]["options"],
        next_step="percakapan"
    )
    # todo:
        # for each message di simpan ke db pesan

@app.route('/percakapan')
def percakapan():

    # expected data diterima:
        # id_percakapan : 24132511532
        # pesan: "ya, benar"
        # context "konfirmasi_nama"

    #terima jawaban
    id_percakapan = request.args.get('id_percakapan')
    nama = request.args.get('nama')
    pesan = request.args.get('pesan')
    context = request.args.get('context')

    print("\n\ndata diterima:")
    print(id_percakapan)
    print(nama)
    print(pesan)

    # tentukan intent jawaban
    if pesan:
        pesan = pesan.lower()
        results = model.predict([bag_of_words(pesan, words)])[0]
        result_index = np.argmax(results)
        tag = labels[result_index]

        print("predict:")
        print(tag)
        print("context:")
        print(context)

        # tentukan kesesuain dengan intent/context & isi data ke formulir

        print("debug:")
        print('context=="konfirmasi_nama" and tag=="affirmative"')
        print(context=="konfirmasi_nama" and tag=="affirmative")

        print('context=="konfirmasi_nik" and tag=="affirmative"')
        print(context=="konfirmasi_nik" and tag=="affirmative")

        print('context=="konfirmasi_nohp" and tag=="affirmative"')
        print(context=="konfirmasi_nohp" and tag=="affirmative"")

        print('context=="konfirmasi_alamat" and tag=="affirmative"')
        print(context=="konfirmasi_alamat" and tag=="affirmative")

        if context=="konfirmasi_nama" and tag=="affirmative" :
            formulir[id_percakapan]['name_correct']=True
        if context=="konfirmasi_nik" and tag=="affirmative" :
            formulir[id_percakapan]['nik_correct']=True
        if context=="konfirmasi_nohp" and tag=="affirmative" :
            formulir[id_percakapan]['nohp_correct']=True
        if context=="konfirmasi_alamat" and tag=="affirmative" :
            formulir[id_percakapan]['alamat_correct']=True        

        # beri pertanyaan selanjutnya
        return jsonify(
            error=False,
            formulir=formulir[id_percakapan],
            id_percakapan=id_percakapan,
            message=["Baiklah", "Benarkah nik kamu %s?" % formulir[id_percakapan]['nik']],
            context="konfirmasi_nik",
            options=pertanyaan["konfirmasi_nik"]["options"],
            next_step="percakapan"
        )

    return jsonify(
        error=True
    )
    # todo
        # gimana cycle pertanyaan nya


@app.route('/tutup_percakapan/:id_percakapan ')
def tutup_percakapan():
    
    # simpan/kirim informasi final

    # beri informasi final & tutup percakapan

    return jsonify(
        error=False,
        formulir=formulir,
        id_percakapan=id_percakapan,
        message=["Baiklah", "Benarkah nik kamu %s?" % nik],
        context="tutup_pecakapan",
        options=[],
        next_step="end"
    )
    # todo:
        # apakah perlu konfirmasi_nama
        # kirim data kemana
        # simpan ke db pelaporan


if __name__ == "__main__":
    app.run()
