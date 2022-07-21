#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip3 install tmdbsimple')
import tmdbsimple as tmdb
import pandas as pd
import time
import itertools


api_key = "d0ccc864ded48afb6a7e28b2d32001ed"
tmdb.API_KEY = api_key


# In[ ]:


df = pd.read_pickle("/content/drive/My Drive/pinky_promise_last.pckl")


# In[ ]:


genres=tmdb.Genres()
list_of_genres=genres.movie_list()['genres']
Genre_ID_to_name={}
for i in range(len(list_of_genres)):
    genre_id=list_of_genres[i]['id']
    genre_name=list_of_genres[i]['name']
    Genre_ID_to_name[genre_id]=genre_name


# In[ ]:


from ast import literal_eval
df.genre_ids=df.genre_ids.apply(literal_eval)


# In[27]:


df.genre_ids[0]


# In[29]:


g = df["genre_ids"]
flat=list(itertools.chain.from_iterable(g))
for i in range(len(list_of_genres)):
  list_of_genres[i]['count'] = flat.count(list_of_genres[i]["id"])

list_of_genres


# In[ ]:


sum=0
for i in range(len(list_of_genres)):
  sum += list_of_genres[i]['count']


# In[31]:


sum


# In[ ]:




