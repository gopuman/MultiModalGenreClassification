#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tmdbsimple')


# In[13]:


#XGB

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import pickle
import tmdbsimple as tmdb

df = pd.DataFrame()
df = pd.read_csv("/content/drive/My Drive/Multimodal Genre Classification/last.csv", engine="python")
 
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

print("SEE",X.shape)


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
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

genre_names = []
for id in df.genre_ids:
  for i in id:
    genre_name=Genre_ID_to_name[i]
    genre_names.append(genre_name)
genre_names = set(genre_names)

#New code for the next review

genre_list=sorted(list(Genre_ID_to_name.keys()))
print(genre_list)

clf = OneVsRestClassifier(XGBClassifier(n_jobs=-1, max_depth=4))

clf.fit(X_train_tfidf, Y_train)
predsnb=clf.predict(X_test_tfidf)

import pickle
f2=open('classifer_xgb','wb')
pickle.dump(clf,f2)
f2.close()

print(classification_report(Y_test, predsnb, target_names=genre_names))

predictionsnb=[]
for i in range(X_test_tfidf.shape[0]):
    pred_genres=[]
    movie_label_scores=predsnb[i]
    for j in range(19):
        #print j
        if movie_label_scores[j]!=0:
            genre=Genre_ID_to_name[genre_list[j]]
            pred_genres.append(genre)
    predictionsnb.append(pred_genres)

for i in range(X_test_tfidf.shape[0]):
    if i%50==0 and i!=0:
        print('MOVIE: ',df.title[test_movies[i]],'\tPREDICTION: ',','.join(predictionsnb[i]))



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


# In[15]:


precs=[]
recs=[]
for i in range(len(test_movies)):
    if i%1==0:
        pos=test_movies[i]
        #test_movie=movies_with_overviews[pos]
        gtids=df.genre_ids[pos]
        gt=[]
        for g in gtids:
            g_name=Genre_ID_to_name[g]
            gt.append(g_name)
#         print predictions[i],movies_with_overviews[i]['title'],gt
        a,b=precision_recall(gt,predictionsnb[i])
        precs.append(a)
        recs.append(b)

print(np.mean(np.asarray(precs)),np.mean(np.asarray(recs)))
print("Average = ",(np.mean(np.asarray(precs))+np.mean(np.asarray(recs)))/2)


# In[ ]:




