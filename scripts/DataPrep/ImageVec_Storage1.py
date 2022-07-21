#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('ls')


# In[ ]:


import os
poster_movies_1 = os.listdir("/content/drive/My Drive/storage1")
len(poster_movies_1)


# In[ ]:


poster_movies = poster_movies_1


# In[ ]:


len(set(poster_movies))


# In[ ]:


poster_movies_copy_1 = [i.replace("_"," ").rstrip(".jpg") for i in poster_movies_1]


# In[ ]:


len(poster_movies_copy_1)


# In[ ]:


get_ipython().system('pip install tensorflow==1.14')


# In[ ]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)


# In[ ]:


import pandas as pd

df = pd.read_csv("/content/drive/My Drive/last.csv",engine="python")
df1 = df[df['title'].isin(poster_movies_copy_1)]
df1 = df1.reset_index()



# In[ ]:


len(df1)


# In[ ]:


imnames=poster_movies_1
feature_list1=[]
genre_list1=[]
file_order1=[]
print("Starting extracting VGG features for scraped images. This will take time, Please be patient...")
print("Total images = ",len(imnames))
failed_files1=[]
succesful_files1=[]
i=0
for mov in range(len(df1)):
    i+=1
    mov_name=df1.title[mov]
    mov_name1=mov_name.replace(':','/')
    poster_name=mov_name.replace(' ','_')+'.jpg'
    if poster_name in imnames:
        img_path="/content/drive/My Drive/storage1/"+poster_name
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            succesful_files1.append(poster_name)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x)
            file_order1.append(img_path)
            feature_list1.append(features)
            genre_list1.append(df1.genre_ids[mov])
            if np.max(np.asarray(feature_list1))==0.0:
                print('problematic',i)
            if i%50==0 or i==1:
                print("Working on Image : ",i)
        except:
            failed_files1.append(poster_name)
            continue
        
    else:
        continue
print("Done with all features, please pickle for future use!")

import pickle
list_pickled=(feature_list1,file_order1,failed_files1,succesful_files1,genre_list1)
f=open('posters_new_features1.pckl','wb')
pickle.dump(list_pickled,f)
f.close()
print("Features dumped to pickle file")

