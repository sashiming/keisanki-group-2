# google colab 使用

!pip install fasttext mecab-python3 unidic-lite

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
data = pd.read_csv('/content/drive/MyDrive/fasttext/wrime-ver2.tsv', delimiter='\t').values.tolist()
output = []
import MeCab
import os
import sys
# from gensim import models
import fasttext
# w2v_model = models.KeyedVectors.load_word2vec_format('../data/jawiki.word_vectors.200d.txt',binary=False)
model = fasttext.load_model('/content/drive/MyDrive/fasttext/cc.ja.300.bin')
def vectorize(sentence):
    sentence_vector = np.zeros(300)
    wakati = MeCab.Tagger("-Owakati")
    monophological = wakati.parse(sentence).split()
    sentence_num = 0
    for word in monophological:
        sentence_num += 1
        word_vec = model[word]
        sentence_vector += word_vec
    return sentence_vector/sentence_num

for i in range(35000):
    list_feeling = []
    sentence = data[i][0]
    list_feeling.append(sentence)
    #関数で処理
    vector = vectorize(sentence)
    list_feeling.append(vector)
    list_emotion = []
    for j in range(4,12):
        emotion_num = ((data[i][j])*3+data[i][j+9]+data[i][j+18]+data[i][j+27])/6
        list_emotion.append(emotion_num)
    list_feeling.append(np.array(list_emotion))
    sentiment_num = ((data[i][12])*3+data[i][21]+data[i][30]+data[i][39])/6
    list_feeling.append(sentiment_num)
    output.append(list_feeling)

np.random.shuffle(output)

valitator_data = pd.DataFrame(output[:5000])
trainer_data = pd.DataFrame(output[5000:])

import pickle
f = open('/content/drive/MyDrive/fasttext/valitator_data.pkl','wb')
g = open('/content/drive/MyDrive/fasttext/trainer_data.pkl','wb')
pickle.dump(valitator_data,f)
pickle.dump(trainer_data,g)
f.close()
g.close()