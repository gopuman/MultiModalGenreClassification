#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tmdbsimple scikit-multilearn')

import pandas as pd
import numpy as np
import pickle
import tmdbsimple as tmdb


# In[2]:


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


# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from skmultilearn.ensemble import RakelD

classifier = RakelD(
    base_classifier=SVC(),
    base_classifier_require_dense=[True, True],
    labelset_size=3
)

classifier.fit(X[msk].toarray(), Y_train)


# In[ ]:


classifier._label_count


# In[ ]:


preds=classifier.predict(X[~msk].toarray())


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(Y_test, preds, target_names=genre_names))


# In[ ]:


preds = preds.toarray()


# In[ ]:


predictions=[]
for i in range(X_test_tfidf.shape[0]):
    pred_genres=[]
    movie_label_scores=preds[i]
    for j in range(19):
        #print j
        if movie_label_scores[j]!=0:
            genre=Genre_ID_to_name[genre_list[j]]
            pred_genres.append(genre)
    predictions.append(pred_genres)

for i in range(X_test_tfidf.shape[0]):
    if i%50==0 and i!=0:
        print('MOVIE: ',df.title[test_movies[i]],'\tPREDICTION: ',','.join(predictions[i]))


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
        a,b=precision_recall(gt,predictions[i])
        precs.append(a)
        recs.append(b)

print(np.mean(np.asarray(precs)),np.mean(np.asarray(recs)))
print("Average = ",(np.mean(np.asarray(precs))+np.mean(np.asarray(recs)))/2)


# In[ ]:




