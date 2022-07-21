#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install urllib3')
import requests
get_ipython().system('pip install tmdbsimple')
import tmdbsimple as tmdb
get_ipython().system('pip install imdbpy')
import imdb
import os
#import urllib2
import requests
import json
import time
import itertools
get_ipython().system('pip install wget')
import wget
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pickle
import pandas as pd


# In[9]:


get_ipython().system('pip install scikit-multilearn')
from skmultilearn.adapt import MLkNN


# In[13]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import pickle
import tmdbsimple as tmdb

df = pd.DataFrame()
df=pd.read_csv("/content/drive/My Drive/Multimodal Genre Classification/last.csv", engine="python")
 
from ast import literal_eval
df.genre_ids=df.genre_ids.apply(literal_eval)

df.genre_ids[0]

api_key = "d0ccc864ded48afb6a7e28b2d32001ed"
tmdb.API_KEY = api_key
#search = tmdb.Search()

genres=tmdb.Genres()
list_of_genres=genres.movie_list()['genres']
Genre_ID_to_name={}
for i in range(len(list_of_genres)):
    genre_id=list_of_genres[i]['id']
    genre_name=list_of_genres[i]['name']
    Genre_ID_to_name[genre_id]=genre_name

# genres=np.zeros((len(top1000_movies),3))
genres=[]
all_ids=[]
for i in range(len(df)):
    id=df.id[i]
    genre_ids=df.genre_ids[i]
    genres.append(genre_ids)
    all_ids.extend(genre_ids)

print(all_ids)

from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()
Y=mlb.fit_transform(genres)

print(Y.shape)
print(np.sum(Y, axis=0))

sample_overview=df.overview[5]
sample_title=df.title[5]
print("The overview for the movie",sample_title," is - \n\n")
print(sample_overview)

from sklearn.feature_extraction.text import CountVectorizer
import re

content=[]
for i in range(len(df)):
    id=df.id[i]
    overview=df.overview[i]
    overview=overview.replace(',','')
    overview=overview.replace('.','')
    content.append(overview)

vectorize=CountVectorizer(max_df=0.95, min_df=0.005)
X=vectorize.fit_transform(content)

print(X.shape)

import pickle
f4=open('X.pckl','wb')
f5=open('Y.pckl','wb')
pickle.dump(X,f4)
pickle.dump(Y,f5)
f6=open('Genredict.pckl','wb')
pickle.dump(Genre_ID_to_name,f6)
f4.close()
f5.close()
f6.close()

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)
print("After",X_tfidf.shape)

msk = np.random.rand(X_tfidf.shape[0]) < 0.8

X_train_tfidf=X_tfidf[msk]
X_test_tfidf=X_tfidf[~msk]
Y_train=Y[msk]
Y_test=Y[~msk]
positions=range(len(df))
# print positions
test_movies=np.asarray(positions)[~msk]
# test_movies


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report

from skmultilearn.adapt import MLkNN

classifier = MLkNN(k=2)
classifier.fit(X_train_tfidf, Y_train)

genre_names = []
for id in df.genre_ids:
  for i in id:
    genre_name=Genre_ID_to_name[i]
    genre_names.append(genre_name)
genre_names = set(genre_names)

genre_list=sorted(list(Genre_ID_to_name.keys()))
print(genre_list)

import pickle
f2=open('classifer_knn','wb')
pickle.dump(classifier,f2)
f2.close()

predstfidf = classifier.predict(X_test_tfidf)

print(classification_report(Y_test, predstfidf, target_names=genre_names))


# In[19]:


predictionsknn=[]
for i in range(X_test_tfidf.shape[0]):
    pred_genres=[]
    movie_label_scores=predstfidf[i]
    for j in range(19):
        #print j
        if movie_label_scores[i][j]!=0:
            genre=Genre_ID_to_name[genre_list[j]]
            pred_genres.append(genre)
    predictionsknn.append(pred_genres)

for i in range(X_test_tfidf.shape[0]):
    if i%50==0 and i!=0:
        print('MOVIE: ',df.title[test_movies[i]],'\tPREDICTION: ',','.join(predictionsknn[i]))


# In[21]:


movie_label_scores[0][1].toarray()


# In[ ]:




