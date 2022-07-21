#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow==1.14')


# In[ ]:


get_ipython().system('ls')


# In[ ]:


import os
poster_movies_3 = os.listdir("/content/drive/My Drive/storage3")
print(len(poster_movies_3))


# In[ ]:


poster_movies = poster_movies_3


# In[ ]:


len(set(poster_movies))


# In[ ]:


poster_movies_copy_3 = [i.replace("_"," ").rstrip(".jpg") for i in poster_movies_3]


# In[ ]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)


# In[ ]:


import pandas as pd

df = pd.read_csv("/content/drive/My Drive/last.csv", engine="python")
df3 = df[df['title'].isin(poster_movies_copy_3)]
df3 = df3.reset_index()


# In[ ]:


len(df3)


# In[ ]:


imnames=poster_movies
feature_list3=[]
genre_list3=[]
file_order3=[]
print("Starting extracting VGG features for scraped images. This will take time, Please be patient...")
print("Total images = ",len(imnames))
failed_files3=[]
succesful_files3=[]
i=0
for mov in range(len(df3)):
    i+=1
    mov_name=df3.title[mov]
    mov_name=mov_name.replace(':','/')
    poster_name=mov_name.replace(' ','_')+'.jpg'
    if poster_name in imnames:
        img_path="/content/drive/My Drive/storage3/"+poster_name
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            succesful_files3.append(poster_name)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x)
            file_order3.append(img_path)
            feature_list3.append(features)
            genre_list3.append(df3.genre_ids[mov])
            if np.max(np.asarray(feature_list3))==0.0:
                print('problematic',i)
            if i%50==0 or i==1:
                print("Working on Image : ",i)
        except:
            failed_files3.append(poster_name)
            continue
        
    else:
        continue
print("Done with all features.)

import pickle
list_pickled=(feature_list3,file_order3,failed_files3,succesful_files3,genre_list3)
f=open('posters_new_features3.pckl','wb')
pickle.dump(list_pickled,f)
f.close()
print("Features dumped to pickle file")


# In[ ]:


len(failed_files3)


# In[ ]:


len(succesful_files3)


# In[ ]:


import pickle
f7=open('/content/drive/My Drive/posters_new_features1.pckl','rb')
list_pickled1=pickle.load(f7)
f7.close()


# In[ ]:


import pickle
f8=open('/content/drive/My Drive/posters_new_features2.pckl','rb')
list_pickled2=pickle.load(f8)
f8.close()


# In[ ]:


a = list_pickled1[0]
b = list_pickled1[1]
c = list_pickled1[2]
d = list_pickled1[3]
e = list_pickled1[4]

a.extend(list_pickled2[0])
b.extend(list_pickled2[1])
c.extend(list_pickled2[2])
d.extend(list_pickled2[3])
e.extend(list_pickled2[4])
print(len(a),len(b),len(c),len(d),len(e))


# In[ ]:


a.extend(feature_list3)
b.extend(file_order3)
c.extend(failed_files3)
d.extend(succesful_files3)
e.extend(genre_list3)


# In[ ]:


print(len(a),len(b),len(c),len(d),len(e))


# In[ ]:


final_feature_list = a
final_files = b
final_failed = c
final_succesful = d
final_genre_list = e


# In[ ]:


import numpy as np
(feature_list,files,failed,succesful,genre_list)=(final_feature_list,final_files,final_failed,final_succesful,final_genre_list)

(a,b,c,d)=feature_list[0].shape
feature_size=a*b*c*d

np_features=np.zeros((len(feature_list),feature_size))
for i in range(len(feature_list)):
    feat=feature_list[i]
    reshaped_feat=feat.reshape(1,-1)
    np_features[i]=reshaped_feat

X=np_features

from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()
Y=mlb.fit_transform(genre_list)

Y.shape


# In[1]:


from keras.models import model_from_json

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# In[ ]:


json_file = open('/content/drive/My Drive/Resnet_Glove.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/drive/My Drive/Resnet_Glove.h5")
print("Loaded model from disk")

