# -*- coding: utf-8 -*-
"""Untitled21.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bGDeL2feStUw260-i3c7i88RY9CrTqJj
"""

!pip install fasttext mecab-python3 unidic-lite

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

import MeCab
import fasttext

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

f = open("/content/drive/MyDrive/fasttext/data.txt","r",encoding = "UTF-8")

data_string = f.read()
f.close()

data = data_string.split()



num_sentence = len(data)

output = []
for i in range(num_sentence):
    sentence = data[i]
    vector = vectorize(sentence)
    output.append(vector)

test_data = pd.DataFrame(output)

test_data.shape

import pickle

h = open("/content/drive/MyDrive/fasttext/test_data.pkl","wb")
pickle.dump(test_data,h)
h.close()