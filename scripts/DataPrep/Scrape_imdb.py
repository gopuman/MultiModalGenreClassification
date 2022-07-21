#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install imdbpy')
import imdb


# In[ ]:


imbd_object = imdb.IMDb()
results = imbd_object.search_movie('Spider-man: Far from home')
movie = results[0]
imbd_object.update(movie)
print("All the information we can get about this movie from IMDB-")
print(movie['genres'])


# In[4]:


x = []
l = [{'id': 10752, 'name': 'War'}, {'id': 18, 'name': 'Drama'}, {'id': 36, 'name': 'History'}, {'id': 28, 'name': 'Action'}]
for i in l:
  x.append(i['name'])
x


# In[ ]:




