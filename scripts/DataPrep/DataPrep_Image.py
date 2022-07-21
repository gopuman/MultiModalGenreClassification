#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv("/content/drive/My Drive/s3.csv")


# In[ ]:


df = df.drop("index", axis=1)


# In[ ]:


df = df.reset_index()


# In[ ]:


df


# In[ ]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pickle
model = VGG16(weights='imagenet', include_top=False)


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
f=open('features_3.pckl','wb')
pickle.dump(feature_list,f)
f.close()


# In[ ]:


failed_files


# In[ ]:


df = df[df.title != "Emma"]


# In[ ]:


(a, b, c, d) = feature_list[0].shape
feature_size = a*b*c*d


# In[ ]:


feature_size


# In[ ]:


np_features=np.zeros((len(feature_list),feature_size))
for i in range(len(feature_list)):
   feat=feature_list[i]
   reshaped_feat=feat.reshape(1,-1)
   np_features[i]=reshaped_feat


# In[ ]:


len(list(np_features))
#list(np_features)


# In[ ]:


df["image_vec"]=list(np_features)


# In[ ]:


df


# In[ ]:


df.to_csv("s3_imagevec.csv")

