#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[6]:


get_ipython().system('pip install tmdbsimple')
import tmdbsimple as tmdb
import os
import pandas as pd
import time


# In[7]:


#initializing the poster directory
poster_folder='posters_final/'
if poster_folder.split('/')[0] in os.listdir('./'):
    print('Folder already exists')
else:
    os.mkdir('./'+poster_folder)


# In[ ]:


#defining the api key and search variables
api_key = "d0ccc864ded48afb6a7e28b2d32001ed"
tmdb.API_KEY = api_key
search = tmdb.Search()


# In[ ]:


#Functions to scrape data
def get_id(movie):
  res = search.movie(query=movie)
  movie_id = res['results'][0]['id']
  return movie_id

def get_info(movie):
  id = get_id(movie)
  movie = tmdb.Movies(id)
  movie_info = movie.info()
  return movie_info

def get_genre(movie):
  l = []
  id = get_id(movie)
  movie = tmdb.Movies(id)
  genres = movie.info()['genres']
  for i in genres:
    l.append(i['name'])
  return l

def get_poster(movie):
  id = get_id(movie)
  movie = tmdb.Movies(id)
  path = movie.info()['poster_path']
  title = movie.info()['original_title']
  url='image.tmdb.org/t/p/original'+path
  title='_'.join(title.split(' '))
  strcmd='wget -O '+'/content/posters_final/'+title+'.jpg '+url
  os.system(strcmd)  
  


# In[ ]:


#Testing the functions
movie = "Spider-man: Far from home"

print(get_id(movie))
info = get_info(movie)
print(get_genre(movie))
#get_poster(movie)
print(info['tagline'])


# In[16]:


genres=tmdb.Genres()
list_of_genres=genres.movie_list()['genres']
print(list_of_genres)


# In[ ]:


def pop_mov(top1000_movies):
  for i in top1000_movies:
    mov.append(i["title"])

def pop_gen(top1000_movies):
  for i in top1000_movies:
    gen.append(get_genre(i["title"]))

def pop_over(top1000_movies):
  for i in top1000_movies:
    over.append(i["overview"])


# In[ ]:


#mov = []
#gen = []
#over = []
#def make_datasets(genre_id):
#  for i in list_of_genres:
#    pop_top1000(i)

def pop_top1000(g):
  disc = tmdb.Discover()
  top1000_movies=[]
  for i in range(1,51):
    print("Iteration-",i)
    if i%15==0:
        time.sleep(7)
    movies_on_this_page = disc.movie(with_genres=str(g),with_original_language="en",page=i)['results']
    top1000_movies.extend(movies_on_this_page)
    pop_mov(top1000_movies)
    pop_gen(top1000_movies)
    pop_over(top1000_movies)


# In[ ]:


disc = tmdb.Discover()
top1000_movies=[]
print('Pulling movie list, Please wait...')
for i in range(1,51):
    if i%15==0:
        time.sleep(7)
    movies_on_this_page=res = disc.movie(with_genres="28",with_original_language="en",page=i)['results']
    top1000_movies.extend(movies_on_this_page)
len(top1000_movies)


# In[ ]:


pop_top1000(list_of_genres[0]['id'])


# In[ ]:


print(len(mov))
print(len(gen))
print(len(over))


# In[20]:


all_movies=tmdb.Movies()
top_movies=all_movies.popular()

# This is a dictionary, and to access results we use the key 'results' which returns info on 20 movies
len(top_movies['results'])
top20_movs=top_movies['results']

all_movies=tmdb.Movies()
top1000_movies=[]
print('Pulling movie list, Please wait...')
for i in range(1,51):
    print("Iteration:",i)
    if i%15==0:
        time.sleep(7)
    movies_on_this_page=all_movies.popular(page=i)['results']
    top1000_movies.extend(movies_on_this_page)
print(top1000_movies)


# In[23]:


print(top1000_movies[0])


# In[ ]:


df = pd.DataFrame(top1000_movies)


# In[25]:


df


# In[ ]:




