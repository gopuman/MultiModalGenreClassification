#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet', include_top=False)


# In[ ]:


import pandas as pd
df = pd.read_pickle("/content/drive/My Drive/pinky_promise_last.pckl")


# In[ ]:


df


# In[ ]:


#imnames=poster_movies_1
feature_list1=[]
genre_list1=[]
file_order1=[]
#print("Starting extracting VGG features for scraped images. This will take time, Please be patient...")
#print("Total images = ",len(imnames))
failed_files1=[]
succesful_files1=[]
i=0
for mov in range(15000, 30000):
    i+=1
    mov_name=df.title[mov]
    mov_name1=mov_name.replace(':','/')
    poster_name=mov_name.replace(' ','_')+'.jpg'
    img_path=df.poster_path[mov]
    try:
      img = image.load_img(img_path, target_size=(224, 224))
      succesful_files1.append(poster_name)
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      features = model.predict(x)
      file_order1.append(img_path)
      feature_list1.append(features)
      genre_list1.append(df.genre_ids[mov])
      if np.max(np.asarray(feature_list1))==0.0:
        print('problematic',i)
      if i%50==0 or i==1:
        print("Working on Image : ",i)
    except:
      failed_files1.append(poster_name)
      continue

import pickle
list_pickled=(feature_list1,file_order1,failed_files1,succesful_files1,genre_list1)
f=open('ResNet_1.pckl','wb')
pickle.dump(list_pickled,f)
f.close()
print("Features dumped to pickle file")


# In[ ]:




