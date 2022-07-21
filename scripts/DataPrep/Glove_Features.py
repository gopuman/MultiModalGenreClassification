#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('unzip "/content/drive/My Drive/glove.6B"')


# In[ ]:


import numpy as np
import os
import pandas as pd


# In[ ]:


glove_dir = './'

embeddings_index = {} #initialize dictionary
f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[ ]:


type(embeddings_index)


# In[ ]:


print(embeddings_index["dog"])


# In[ ]:


get_ipython().system('pip install stop_words')


# In[ ]:


from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
tokenizer = RegexpTokenizer(r'\w+')

en_stop = get_stop_words('en')


# In[ ]:


len(en_stop)


# In[ ]:


df=pd.read_pickle("/content/drive/My Drive/pinky_promise_last.pckl")


# In[ ]:


from ast import literal_eval
df.genre_ids=df.genre_ids.apply(literal_eval)


# In[ ]:


movie_mean_wordvec=np.zeros((len(df),300))
movie_mean_wordvec.shape


# In[ ]:


genres=[]
rows_to_delete=[]
for i in range(len(df)):
    movie_genres=df.genre_ids[i]
    genres.append(movie_genres)
    overview=df.overview[i]
    tokens = tokenizer.tokenize(overview)
    stopped_tokens = [k for k in tokens if not k in en_stop]
    count_in_vocab=0
    s=0
    if len(stopped_tokens)==0:
        rows_to_delete.append(i)
        genres.pop(-1)
#         print(overview)
#         print("sample ",i,"had no nonstops")
    else:
        for tok in stopped_tokens:
            if tok.lower() in embeddings_index:
                count_in_vocab+=1
                s+=embeddings_index[tok.lower()]
        if count_in_vocab!=0:
            movie_mean_wordvec[i]=s/float(count_in_vocab)
        else:
            rows_to_delete.append(i)
            genres.pop(-1)
#             print(overview)
#             print("sample ",i,"had no word2vec")


# In[ ]:


mask2=[]
for row in range(len(movie_mean_wordvec)):
    if row in rows_to_delete:
        mask2.append(False)
    else:
        mask2.append(True)


# In[ ]:


X=movie_mean_wordvec[mask2]


# In[ ]:


X.shape


# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

Y=mlb.fit_transform(genres)
Y.shape


# In[ ]:


type(genres)


# In[ ]:


list(mlb.classes_)


# In[ ]:


(X,Y)=glove_features

print(X.shape)
print(Y.shape)


# In[ ]:


import numpy as np
mask_text=np.random.rand(len(X))<0.8

X_train=X[mask_text]
Y_train=Y[mask_text]
X_test=X[~mask_text]
Y_test=Y[~mask_text]


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Activation

model_textual = Sequential([
    Dense(300, input_shape=(300,)),
    Activation('relu'),
    Dense(14),
    Activation('softmax'),
])

model_textual.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model_textual.fit(X_train, Y_train, epochs=10000, batch_size=500, verbose=0 )


# In[ ]:


score = model_textual.evaluate(X_test, Y_test, batch_size=249)
print(score)


# In[ ]:


print("%s: %.2f%%" % (model_textual.metrics_names[1], score[1]*100))


# In[ ]:


Y_preds=model_textual.predict(X_test)


# In[ ]:


get_ipython().system('pip install tmdbsimple')
import tmdbsimple as tmdb


# In[ ]:


api_key = "d0ccc864ded48afb6a7e28b2d32001ed"
tmdb.API_KEY = api_key
genres=tmdb.Genres()
list_of_genres=genres.movie_list()['genres']
Genre_ID_to_name={}
for i in range(len(list_of_genres)):
  genre_id=list_of_genres[i]['id']
  genre_name=list_of_genres[i]['name']
  Genre_ID_to_name[genre_id]=genre_name


# In[ ]:


genre_list=sorted(list(Genre_ID_to_name.keys()))


# In[ ]:


def precision_recall(gt,preds):
   TP=0
   FP=0
   FN=0
   for t in gt:
       if t in preds:
           TP+=1
       else:
           FN+=1
   for p in preds:
       if p not in gt:
           FP+=1
   if TP+FP==0:
       precision=0
   else:
       precision=TP/float(TP+FP)
   if TP+FN==0:
       recall=0
   else:
       recall=TP/float(TP+FN)
   return precision,recall


# In[ ]:


print("Our predictions for the movies are - \n")
precs=[]
recs=[]
for i in range(len(Y_preds)):
    row=Y_preds[i]
    gt_genres=Y_test[i]
    gt_genre_names=[]
    for j in range(14):
        if gt_genres[j]==1:
            gt_genre_names.append(Genre_ID_to_name[genre_list[j]])
    top_3=np.argsort(row)[-3:]
    predicted_genres=[]
    for genre in top_3:
        predicted_genres.append(Genre_ID_to_name[genre_list[genre]])
    (precision,recall)=precision_recall(gt_genre_names,predicted_genres)
    precs.append(precision)
    recs.append(recall)
    if i%50==0:
        print("Predicted: ",predicted_genres," Actual: ",gt_genre_names)


# In[ ]:


print(np.mean(np.asarray(precs)),np.mean(np.asarray(recs)))
print("avg=",(np.mean(np.asarray(precs))+np.mean(np.asarray(recs)))/2)


# In[ ]:


print("f1-score", (2*(np.mean(np.asarray(precs)) * np.mean(np.asarray(recs)))) / (np.mean(np.asarray(precs)) + np.mean(np.asarray(recs))))

