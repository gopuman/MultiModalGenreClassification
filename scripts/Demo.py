#!/usr/bin/env python
# coding: utf-8

# In[155]:


get_ipython().system('pip install tmdbsimple')
import tmdbsimple as tmdb
from IPython.display import Image


api_key = "4c8081ac8d03d88eb6332eabb7f3d950"
tmdb.API_KEY = api_key #This sets the API key setting for the tmdb object
search = tmdb.Search()

def get_movie_info_tmdb(movie):
    response = search.movie(query=movie)
    id=response['results'][0]['id']
    movie = tmdb.Movies(id)
    info=movie.info()
    return info
  
def grab_poster_tmdb(movie):
    response = search.movie(query=movie)
    id=response['results'][0]['id']
    movie = tmdb.Movies(id)
    posterp=movie.info()['poster_path']
    title=movie.info()['original_title']
    url='image.tmdb.org/t/p/original'+posterp
    title='_'.join(title.split(' '))
    strcmd='wget -O poster.jpg '+url
    os.system(strcmd)

def get_movie_genres_tmdb(movie):
    response = search.movie(query=movie)
    id=response['results'][0]['id']
    movie = tmdb.Movies(id)
    genres=movie.info()['genres']
    return genres

def genlist(g):
  l=[]
  for i in g:
    l.append(i['id'])
  return l

#Import model
from keras.models import model_from_json

json_file = open('/content/drive/My Drive/Resnet_Glove.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/drive/My Drive/Resnet_Glove.h5")
# print("Loaded model from disk")

from keras import optimizers
opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
loaded_model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

res = ResNet50(weights='imagenet')

get_ipython().system('unzip "/content/drive/My Drive/glove.6B"')

import os
glove_dir = './'

embeddings_index = {} #initialize dictionary
f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# !pip install stop_words

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
tokenizer = RegexpTokenizer(r'\w+')

en_stop = get_stop_words('en')


# In[184]:


inp = input("ENTER THE MOVIE TO BE PREDICTED : ")
grab_poster_tmdb(inp)
overview = get_movie_info_tmdb(inp)['overview']
print(overview)
y = get_movie_genres_tmdb(inp)
actual = genlist(y)
actual.sort()

#Resnet Features
img = image.load_img('/content/poster.jpg',target_size=(224,224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

features = res.predict(img)

#Glove Features
tokens = tokenizer.tokenize(overview)
stopped_tokens = [k for k in tokens if not k in en_stop]
count_in_vocab=0
s=0
for tok in stopped_tokens:
  if tok.lower() in embeddings_index:
    count_in_vocab+=1
    s+=embeddings_index[tok.lower()]
  if count_in_vocab!=0:
    movie_mean_wordvec=s/float(count_in_vocab)

#Joining
x1 = features.flatten()
x2 = movie_mean_wordvec.flatten()
x3 = np.concatenate((x1,x2),axis=0)

#Predict
pred = loaded_model.predict(np.expand_dims(x3, 0))

import pickle
f6=open('/content/drive/My Drive/Genredict.pckl','rb')
Genre_ID_to_name=pickle.load(f6)
f6.close()
genre_list=sorted(list(Genre_ID_to_name.keys()))

import numpy as np

precs=[]
recs=[]
for i in range(1):
    row=pred[0]
    gt_genres = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for k in actual:
      gt_genres[genre_list.index(k)] = 1
    gt_genre_names=[]
    for j in range(14):
        if gt_genres[j]==1:
            gt_genre_names.append(Genre_ID_to_name[genre_list[j]])
    top_3=np.argsort(row)[-3:]
    predicted_genres=[]
    for genre in top_3:
        predicted_genres.append(Genre_ID_to_name[genre_list[genre]])
    print("Predicted: ",','.join(predicted_genres)," Actual: ",','.join(gt_genre_names))
    print(top_3)


# In[ ]:




