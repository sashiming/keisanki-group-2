import pandas as pd
import numpy as np
data = pd.read_csv('../data/wrime-ver2.tsv', delimiter='\t').values.tolist()
output = []
import MeCab
import os
import sys
# from gensim import models
import fasttext
# w2v_model = models.KeyedVectors.load_word2vec_format('../data/jawiki.word_vectors.200d.txt',binary=False)
model = fasttext.load_model('../data/cc.ja.300.bin')
def vectorize(sentence):
    sentence_vector = np.zeros(200)
    wakati = MeCab.Tagger("-Owakati")
    monophological = wakati.parse(sentence).split()
    sentence_num = 0
    for word in monophological:
        if w2v_model.wv.__contains__(word):
            sentence_num += 1
            word_vec = w2v_model.wv.__getitem__(word)
            sentence_vector += word_vec
        else:
            sentence_num += 1
            word_similar = w2v_model.wv.most_similar(word)
            word_vec = w2v_model.wv.__getitem__(word_similar)
            sentence_vector += word_vec
    return sentence_vector/sentence_num
    
# print(vectorize('私は猫が好きです'))

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
f = open('../data/valitator_data.pkl','wb')
g = open('../data/trainer_data.pkl','wb')
pickle.dump(valitator_data,f)
pickle.dump(trainer_data,g)
f.close()
g.close()

