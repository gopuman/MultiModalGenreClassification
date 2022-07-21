#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tmdbsimple')
import pandas as pd
import numpy as np
import pickle
import tmdbsimple as tmdb


# In[3]:


df = pd.DataFrame()
df = pd.read_csv("/content/drive/My Drive/last.csv", engine="python")

from ast import literal_eval
df.genre_ids=df.genre_ids.apply(literal_eval)

api_key = "d0ccc864ded48afb6a7e28b2d32001ed"
tmdb.API_KEY = api_key

genres=tmdb.Genres()
list_of_genres=genres.movie_list()['genres']
Genre_ID_to_name={}
for i in range(len(list_of_genres)):
    genre_id=list_of_genres[i]['id']
    genre_name=list_of_genres[i]['name']
    Genre_ID_to_name[genre_id]=genre_name

genres=[]
all_ids=[]
for i in range(len(df)):
    id=df.id[i]
    genre_ids=df.genre_ids[i]
    genres.append(genre_ids)
    all_ids.extend(genre_ids)

from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()
Y=mlb.fit_transform(genres)

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

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)

msk = np.random.rand(X_tfidf.shape[0]) < 0.8

X_train_tfidf=X_tfidf[msk]
X_test_tfidf=X_tfidf[~msk]
Y_train=Y[msk]
Y_test=Y[~msk]

positions=range(len(df))
# print positions
test_movies=np.asarray(positions)[~msk]

genre_names = []
for id in df.genre_ids:
  for i in id:
    genre_name=Genre_ID_to_name[i]
    genre_names.append(genre_name)
genre_names = set(genre_names)

genre_list=sorted(list(Genre_ID_to_name.keys()))
print(genre_list)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

classifnb = OneVsRestClassifier(MultinomialNB())
classifknn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=1))
classifdt = OneVsRestClassifier(DecisionTreeClassifier())

classifiers = [('nb',classifnb),('knn',classifknn),('dt',classifdt)]
#vc = VotingClassifier(estimators=classifiers,voting='soft')



# In[ ]:


classifnb.fit(X[msk].toarray(), Y_train)
classifknn.fit(X[msk].toarray(), Y_train)
classifdt.fit(X[msk].toarray(), Y_train)


# In[ ]:




