#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install tmdbsimple')
import tmdbsimple as tmdb
import pandas as pd
import time

api_key = "d0ccc864ded48afb6a7e28b2d32001ed"
tmdb.API_KEY = api_key
#search = tmdb.Search()

top1000_movies=[]

def pop(j):
    disc = tmdb.Discover()
    for i in range(1,501):
        print(j,"---",i)
        if i%15==0:
            time.sleep(7)
        movies_on_this_page=disc.movie(with_genres=str(j),with_original_language="en",page=i)['results']
        for k in movies_on_this_page:
            if k:
                top1000_movies.extend(movies_on_this_page)

l = [28, 12, 16, 35, 80, 99, 18, 10751, 14, 36, 27, 10402, 9648, 10749, 878, 10770, 53, 10752, 37]

for i in l:
    pop(i)

print(len(top1000_movies))


#df = pd.DataFrame(top1000_movies)

#df.to_csv("genre.csv")

'''
all_movies=tmdb.Movies()
top_movies=all_movies.popular()

# This is a dictionary, and to access results we use the key 'results' which returns info on 20 movies
len(top_movies['results'])
top20_movs=top_movies['results']
'''


# In[ ]:


res = list(filter(None,top1000_movies))
df = pd.DataFrame(res)


# In[ ]:


df = df.drop(['popularity'], axis = 1) 
df = df.drop(['vote_count'], axis = 1) 
df = df.drop(['video'], axis = 1) 
df = df.drop(['adult'], axis = 1) 
df = df.drop(['backdrop_path'], axis = 1) 
df = df.drop(['original_title'], axis = 1) 
df = df.drop(['vote_average'], axis = 1) 
df = df.drop(['release_date'], axis = 1) 

df1 = pd.DataFrame()
df1 = df
df1


# In[ ]:


df = df.drop_duplicates(subset="title",keep="first",inplace=False)


# In[ ]:


df.to_csv("final_dataset.csv")


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


df = pd.read_csv("/content/drive/My Drive/final_dataset.csv",engine="python")


# In[ ]:


def list2pairs(l):
    # itertools.combinations(l,2) makes all pairs of length 2 from list l.
    pairs = list(itertools.combinations(l, 2))
    # then the one item pairs, as duplicate pairs aren't accounted for by itertools
    for i in l:
        pairs.append([i,i])
    return pairs


# In[ ]:


genres=tmdb.Genres()
list_of_genres=genres.movie_list()['genres']
Genre_ID_to_name={}
for i in range(len(list_of_genres)):
    genre_id=list_of_genres[i]['id']
    genre_name=list_of_genres[i]['name']
    Genre_ID_to_name[genre_id]=genre_name


# In[5]:


import itertools
import numpy as np

allPairs = []
for movie in res:
    allPairs.extend(list2pairs(movie['genre_ids']))
    
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


# In[14]:


g = list(df["genre_ids"])
flat=list(itertools.chain.from_iterable(g))
for i in range(len(list_of_genres)):
  list_of_genres[i]['count'] = flat.count(list_of_genres[i]["id"])

list_of_genres


# In[ ]:


top1000_movies=[]

def pop(j):
    disc = tmdb.Discover()
    print('Pulling movie list, Please wait...')
    for i in range(151,201):
        print("iteration:",i)
        if i%15==0:
            time.sleep(7)
        movies_on_this_page = disc.movie(with_genres=str(j),with_original_language="en",page=i)['results']
        top1000_movies.extend(movies_on_this_page)

#l = [28, 12, 16, 35, 80, 99, 10751, 14, 36, 27, 10402, 9648, 10749, 878, 10770, 53, 10752, 37]
l=[80]

for i in l:
    pop(i)

df2 = pd.DataFrame(top1000_movies)

def fun(x):
  if 35 in x or 28 in x or 18 in x:
    return False
  else:
    return True

df2 = df2[df2['genre_ids'].map(fun)]

df2 = df2.drop(['popularity'], axis = 1) 
df2 = df2.drop(['vote_count'], axis = 1) 
df2 = df2.drop(['video'], axis = 1) 
df2 = df2.drop(['adult'], axis = 1) 
df2= df2.drop(['backdrop_path'], axis = 1) 
df2 = df2.drop(['original_title'], axis = 1) 
df2 = df2.drop(['vote_average'], axis = 1) 
df2 = df2.drop(['release_date'], axis = 1) 


# In[ ]:


df = df1.append(df2,sort=True)


# In[ ]:


df


# In[ ]:


df1.drop_duplicates(subset ="title", 
                     keep = False, inplace = True) 
len(df1)


# In[ ]:


df1.to_csv("dataset.csv")


# In[ ]:




