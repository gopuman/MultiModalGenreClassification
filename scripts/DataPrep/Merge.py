#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
get_ipython().system('pip install tmdbsimple')
import tmdbsimple as tmdb


# In[ ]:


api_key = "d0ccc864ded48afb6a7e28b2d32001ed"
tmdb.API_KEY = api_key


# In[ ]:


df = pd.read_csv("last.csv", engine="python")


# In[40]:


df['genre_ids'][0]
len(df)


# In[41]:


df = df.drop_duplicates(subset="title",keep="first",inplace=False)
len(df)


# In[42]:


df = df.dropna()
len(df)


# In[ ]:


from ast import literal_eval
df.genre_ids = df.genre_ids.apply(literal_eval)


# In[ ]:


df.to_csv("last.csv")


# In[ ]:


df = pd.read_csv("last.csv")


# In[ ]:


df.genre_ids[0]


# In[ ]:


df.to_csv("latest.csv")


# In[ ]:


df = pd.read_csv("latest.csv")


# In[ ]:


del df["Unnamed: 0"]


# In[ ]:


df.genre_ids[89]


# In[44]:


import itertools

genres=tmdb.Genres()
list_of_genres=genres.movie_list()['genres']

g = list(df["genre_ids"])
flat=list(itertools.chain.from_iterable(g))

for i in range(len(list_of_genres)):
  list_of_genres[i]['count'] = flat.count(list_of_genres[i]["id"])

list_of_genres


# In[51]:


gencount = []
gennames = []
for i in range(19):
  gencount.append(list_of_genres[i]['count'])
  gennames.append(list_of_genres[i]['name'])

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()    
y_pos = np.arange(len(gennames))
performance = gencount

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, gennames,rotation=90)

plt.show()


# In[ ]:


gencount


# In[ ]:


df.genre_ids[0]


# In[ ]:


mask = df.genre_ids.apply(lambda x: 28 in x)
df1 = df[mask]
print (df1)



# In[ ]:


movie_list = df.values.tolist()
movie_list


# In[ ]:


genres=tmdb.Genres()
list_of_genres=genres.movie_list()['genres']
print(list_of_genres)

Genre_ID_to_name={}
for i in range(len(list_of_genres)):
    genre_id=list_of_genres[i]['id']
    genre_name=list_of_genres[i]['name']
    Genre_ID_to_name[genre_id]=genre_name


# In[ ]:


def list2pairs(l):
    # itertools.combinations(l,2) makes all pairs of length 2 from list l.
    pairs = list(itertools.combinations(l, 2))
    # then the one item pairs, as duplicate pairs aren't accounted for by itertools
    for i in l:
        pairs.append([i,i])
    return pairs


# In[ ]:


import itertools
import numpy as np

allPairs = []
for i in range(len(df)):
    allPairs.extend(list2pairs(df.genre_ids[i]))
    
nr_ids = np.unique(allPairs)
visGrid = np.zeros((len(nr_ids), len(nr_ids)))
for p in allPairs:
    visGrid[np.argwhere(nr_ids==p[0]), np.argwhere(nr_ids==p[1])]+=1
    if p[1] != p[0]:
        visGrid[np.argwhere(nr_ids==p[1]), np.argwhere(nr_ids==p[0])]+=1


# In[ ]:


print(visGrid.shape)
print(len(Genre_ID_to_name.keys()))


# In[ ]:


import seaborn as sns
annot_lookup = []
for i in range(len(nr_ids)):
    annot_lookup.append(Genre_ID_to_name[nr_ids[i]])

sns.heatmap(visGrid, xticklabels=annot_lookup, yticklabels=annot_lookup)


# In[ ]:


print(df.loc[1589])


# In[ ]:


#Use df1 for mods
df1 = df


# In[ ]:


df1.info()


# In[ ]:


x = pd.DataFrame



# In[ ]:




