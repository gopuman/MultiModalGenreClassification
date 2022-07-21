#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tmdbsimple')
get_ipython().system('pip install stop_words')


# In[ ]:


import pandas as pd

df = pd.read_pickle("/content/drive/My Drive/promise_last.pckl")


# In[ ]:


len(df)


# In[ ]:


from gensim import models
# model2 = models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True) 
model2 = models.KeyedVectors.load_word2vec_format('/content/drive/My Drive/GoogleNews-vectors-negative300.bin', binary=True)


# In[ ]:


from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
tokenizer = RegexpTokenizer(r'\w+')

en_stop = get_stop_words('en')


# In[ ]:


len(en_stop)


# In[ ]:


import numpy as np

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


rows_to_delete
#df.title[11043]
#df.overview[11043]


# In[ ]:


df = df[df.title!="Bright 2"]


# In[ ]:


df = df[df.title!="Fuggy Fuggy"]


# In[ ]:


len(df)


# In[ ]:


df = df.reset_index()


# In[ ]:


df


# In[ ]:


df = df.drop("Unnamed: 0",axis=1)
df = df.drop("index",axis=1)


# In[ ]:


df


# In[ ]:


mask2=[]
for row in range(len(movie_mean_wordvec)):
    if row in rows_to_delete:
        mask2.append(False)
    else:
        mask2.append(True)


# In[ ]:


X = movie_mean_wordvec[mask2]


# In[ ]:


X.shape


# In[ ]:


text_vec = []


# In[ ]:


for i in range(len(X)):
  text_vec.append(X[i])


# In[ ]:


df.drop("text_vec",axis=1)


# In[ ]:


df['text_vec'] = text_vec


# In[ ]:


df


# In[ ]:


type(df.image_vec[0])


# In[ ]:


df.to_pickle("pinky_promise_last.pckl")


# In[ ]:


dfx = pd.read_pickle("pinky_promise_last.pckl")


# In[ ]:


dfx


# In[ ]:


type(df.image_vec[0])

