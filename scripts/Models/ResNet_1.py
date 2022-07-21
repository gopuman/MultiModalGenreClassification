#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')


# In[ ]:


import pandas as pd
df1 = pd.read_pickle("/content/drive/My Drive/pinky_promise_last.pckl")


# In[ ]:


#imnames=poster_movies_1
feature_list1=[]
genre_list1=[]
file_order1=[]
#print("Total images = ",len(imnames))
failed_files1=[]
succesful_files1=[]
i=0
for mov in range(0,15000):
    i+=1
    mov_name=df1.title[mov]
    mov_name1=mov_name.replace(':','/')
    poster_name=mov_name.replace(' ','_')+'.jpg'
    img_path=df1.poster_path[mov]
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
        
import pickle
list_pickled=(feature_list1,file_order1,failed_files1,succesful_files1,genre_list1)
f=open('Resnet_features_1.pckl','wb')
pickle.dump(list_pickled,f)
f.close()
print("Features dumped to pickle file")


# In[ ]:


len(feature_list1)


# In[ ]:


feature_list1[0].shape


# In[ ]:


(a,b)=feature_list1[0].shape
feature_size=a*b

np_features=np.zeros((len(feature_list1),feature_size))
for i in range(len(feature_list1)):
    feat=feature_list1[i]
    reshaped_feat=feat.reshape(1,-1)
    np_features[i]=reshaped_feat


# In[ ]:


import pickle

f1 = open("/content/drive/My Drive/Resnet_features_1.pckl","rb")
x1 = pickle.load(f1)


# In[11]:


len(x2)


# In[ ]:


f2 = open("/content/drive/My Drive/Resnet_features_2.pckl","rb")
x2 = pickle.load(f2)


# In[ ]:


import numpy as np
X = np.concatenate((x1,x2),axis=0)


# In[ ]:


f3 = open("/content/drive/My Drive/Resnet_features_3.pckl","rb")
x3 = pickle.load(f3)


# In[ ]:


import numpy as np
X = np.concatenate((X,x3),axis=0)


# In[9]:


len(X)

