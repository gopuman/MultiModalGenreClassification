#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv("/content/drive/My Drive/last.csv",engine="python")


# In[3]:


from gensim import models
# model2 = models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
model2 = models.KeyedVectors.load_word2vec_format('/content/drive/My Drive/GoogleNews-vectors-negative300.bin', binary=True)


# In[4]:


print(model2['king'].shape)
print(model2['dog'].shape)


# In[5]:


get_ipython().system('pip install stop_words')


# In[ ]:


from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
tokenizer = RegexpTokenizer(r'\w+')

en_stop = get_stop_words('en')


# In[7]:


len(en_stop)


# In[8]:


import numpy as np

movie_mean_wordvec=np.zeros((len(df),300))
movie_mean_wordvec.shape


# In[ ]:


from ast import literal_eval
df.genre_ids=df.genre_ids.apply(literal_eval)


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
            if tok.lower() in model2.vocab:
                count_in_vocab+=1
                s+=model2[tok.lower()]
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


# In[13]:


X.shape


# In[14]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

Y=mlb.fit_transform(genres)
Y.shape


# In[ ]:


import pickle

textual_features=(X,Y)
f9=open('textual_features.pckl','wb')
pickle.dump(textual_features,f9)
f9.close()


# In[ ]:


import pickle
# textual_features=(X,Y)
f9=open('/content/drive/My Drive/textual_features.pckl','rb')
textual_features=pickle.load(f9)
f9.close()


# In[17]:


(X,Y)=textual_features

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


# In[23]:


model_textual.fit(X_train, Y_train, epochs=10, batch_size=500)


# In[24]:


model_textual.fit(X_train, Y_train, epochs=10000, batch_size=500,verbose = 0)


# In[25]:


score = model_textual.evaluate(X_test, Y_test, batch_size=249)
print(score)


# In[26]:


print("%s: %.2f%%" % (model_textual.metrics_names[1], score[1]*100))


# In[ ]:


Y_preds=model_textual.predict(X_test)


# In[28]:


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


# In[33]:


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


# In[34]:


print(np.mean(np.asarray(precs)),np.mean(np.asarray(recs)))
print("avg=",(np.mean(np.asarray(precs))+np.mean(np.asarray(recs)))/2)


# In[37]:


from keras.models import model_from_json

model_json = model_textual.to_json()
with open("deep_text.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_textual.save_weights("deep_text.h5")
print("Saved model to disk")


# In[39]:


json_file = open('/content/drive/My Drive/deep_text.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new modelyjrtur
loaded_model.load_weights("/content/drive/My Drive/deep_text.h5")
print("Loaded model from disk")


# In[40]:


loaded_model


# In[ ]:


loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# In[ ]:


score = loaded_model.evaluate(X_train, Y_train, verbose=0)


# In[44]:


loaded_model.fit(X_train, Y_train, epochs=10, batch_size=500)


# In[ ]:




