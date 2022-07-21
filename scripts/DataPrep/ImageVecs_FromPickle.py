#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv("/content/drive/My Drive/Sol3_final.csv")


# In[ ]:


df


# In[ ]:


df["poster_path"] = "NaN"


# In[ ]:


import os
storage1 = os.listdir("/content/drive/My Drive/storage1")


# In[ ]:


storage2 = os.listdir("/content/drive/My Drive/storage2")
storage3 = os.listdir("/content/drive/My Drive/storage3")


# In[ ]:


len(storage3)


# In[ ]:


storage1_movs = []


# In[ ]:


storage2_movs = []
storage3_movs = []


# In[ ]:


for x in storage1:
  x = x.rstrip('(1).jpg')
  x = ' '.join(x.split("_"))
  x = x.strip()
  storage1_movs.append(x)


# In[ ]:


for x in storage2:
  x = x.rstrip('(1).jpg')
  x = ' '.join(x.split("_"))
  x = x.strip()
  storage2_movs.append(x)


# In[ ]:


for x in storage3:
  x = x.rstrip('(1).jpg')
  x = ' '.join(x.split("_"))
  x = x.strip()
  storage3_movs.append(x)


# In[ ]:


storage1_movs = list(set(storage1_movs))


# In[ ]:


storage2_movs = list(set(storage2_movs))
storage3_movs = list(set(storage3_movs))


# In[ ]:


def update():
  for i in range(len(df)):
    if(df.title[i] in storage3_movs):
      mov = df.title[i].replace(" ","_")
      path = "/content/drive/My Drive/storage3/"+mov+".jpg"
      df.poster_path[i] = path


# In[ ]:


update()


# In[ ]:


s = storage1[0]


# In[ ]:


s = s.rstrip('.jpg')
s = ' '.join(s.split("_"))
s = s.strip()
s


# In[ ]:


s = "King Dave"
mov = s.replace(" ","_")
path = "/content/drive/My Drive/storage1/"+mov+".jpg"
path


# In[ ]:


df = df[df.poster_path!="NaN"]


# In[ ]:


len(df)


# In[ ]:


df.to_csv("Sol3_final.csv")


# In[ ]:


import re


# In[ ]:


df


# In[ ]:


s1 = df[df['poster_path'].str.match('/content/drive/My Drive/storage1')]
s2 = df[df['poster_path'].str.match('/content/drive/My Drive/storage2')]
s3 = df[df['poster_path'].str.match('/content/drive/My Drive/storage3')]


# In[ ]:


len(s1)


# In[ ]:


s1.to_csv("s1.csv")
s2.to_csv("s2.csv")
s3.to_csv("s3.csv")


# In[ ]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pickle
model = VGG16(weights='imagenet', include_top=False)


# In[ ]:


df = pd.read_csv("s1.csv")


# In[ ]:


df = df.reset_index()


# In[ ]:


df = df.drop("index",axis=1)
df = df.drop("Unnamed: 0",axis=1)
df = df.drop("Unnamed: 0.1",axis=1)
df = df.drop("Unnamed: 0.1.1",axis=1)


# In[ ]:


df


# In[ ]:


#allnames=os.listdir(poster_folder)
#imnames=[j for j in allnames if j.endswith('.jpg')]
feature_list=[]
genre_list=[]
#file_order=[]
#print "Starting extracting VGG features for scraped images. This will take time, Please be patient..."
#print "Total images = ",len(imnames)
failed_files=[]
#succesful_files=[]
i=0
for mov in range(len(df)):
   i+=1
   # mov_name=mov['original_title']
   # mov_name1=mov_name.replace(':','/')
   # poster_name=mov_name.replace(' ','_')+'.jpg'
   #if poster_name in imnames:
   #imname =
   img_path=df.poster_path[mov]
   try:
     img = image.load_img(img_path, target_size=(224, 224))
     #succesful_files.append(imname)
     x = image.img_to_array(img)
     x = np.expand_dims(x, axis=0)
     x = preprocess_input(x)
     features = model.predict(x)
     #file_order.append(img_path)
     feature_list.append(features)
     genre_list.append(df.genre_ids[mov])
     if np.max(np.asarray(feature_list))==0.0:
       print('problematic',i)
     if i%50==0 or i==1:
       print("Working on Image : ",i)
       
   except:
     failed_files.append(df.title[mov])
     continue


# In[ ]:


import pickle
f=open('features_1.pckl','wb')
pickle.dump(feature_list,f)
f.close()


# In[ ]:


len(failed_files)


# In[ ]:


len(feature_list)


# In[ ]:


len(df)


# In[ ]:


failed_files


# In[ ]:


df = df[df.title!="Now You See It"]


# In[ ]:


df


# In[ ]:


(a,b,c,d)=feature_list[0].shape
feature_size=a*b*c*d


# In[ ]:


feature_size


# In[ ]:


np_features=np.zeros((len(feature_list),feature_size))
for i in range(len(feature_list)):
    feat=feature_list[i]
    reshaped_feat=feat.reshape(1,-1)
    np_features[i]=reshaped_feat


# In[ ]:


len(np_features)


# In[ ]:


len(df)


# In[ ]:


df["image_vec"] = list(np_features)


# In[ ]:


df


# In[ ]:


df.to_csv("s1_imagevec.csv")


# In[ ]:




