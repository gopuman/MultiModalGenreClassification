#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
poster_movies_1 = os.listdir("/content/drive/My Drive/storage1")
poster_movies_2 = os.listdir("/content/drive/My Drive/storage2")
poster_movies_3 = os.listdir("/content/drive/My Drive/storage3")


# In[ ]:


poster_movies = poster_movies_1 + poster_movies_2 + poster_movies_3


# In[ ]:


len(poster_movies)


# In[ ]:


poster_movies = list(set(poster_movies))


# In[ ]:


len(poster_movies)


# In[ ]:


movs = []


# In[ ]:


for x in poster_movies:
  x = x.rstrip("(1).jpg")
  x = ' '.join(x.split("_"))
  x = x.strip()
  movs.append(x)


# In[ ]:


movs


# In[ ]:


s = 'Six Pack.jpg'
s = s.rstrip("(1).jpg")
s


# In[ ]:


movs[378] = movs[378].rstrip("(2")


# In[ ]:


movs[378].strip()


# In[ ]:


import pandas as pd
df = pd.DataFrame()
df['title'] = movs


# In[ ]:


df.title[16]


# In[ ]:


last = pd.read_csv("/content/drive/My Drive/last.csv",engine="python")


# In[ ]:


overviews = []
not_found = []


# In[ ]:


m = df.title[16]
x = last[last.title==m].index.values.astype(int)[0]
last.overview[x]


# In[ ]:


for i in range(len(df)):
  m = df.title[i]
  try:
    x = last[last.title==m].index.values.astype(int)[0]
    overviews.append(last.overview[x])
  except:
    not_found.append(m)
    overviews.append("NaN")


# In[ ]:


len(not_found)


# In[ ]:


len(overviews)


# In[ ]:


df['overview'] = overviews


# In[ ]:


df


# In[ ]:


df2 = df


# In[ ]:


for i in range(len(df)):
  if df.overview[i] == "NaN":
    df2 = df.drop(df.index[i])


# In[ ]:


df = df[df.overview!="NaN"]


# In[ ]:


df.title[15]


# In[ ]:


df = df.reset_index()


# In[ ]:


df.title[16]


# In[ ]:


gen = []
not_found = []


# In[ ]:


for i in range(len(df)):
  m = df.title[i]
  try:
    x = last[last.title==m].index.values.astype(int)[0]
    gen.append(last.genre_ids[x])
  except:
    not_found.append(m)
    gen.append("NaN")


# In[ ]:


df["genre_ids"] = gen


# In[ ]:


df


# In[ ]:


df = df.drop("index",axis=1)
df = df.drop("level_0",axis=1)


# In[ ]:


df


# In[ ]:


df.to_csv()


# In[ ]:


df


# In[ ]:


df.to_csv("sol3.csv")


# In[ ]:




