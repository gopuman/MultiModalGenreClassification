#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_pickle("/content/drive/My Drive/pinky_promise_last.pckl")


# In[ ]:


df


# In[ ]:


from ast import literal_eval
df.genre_ids=df.genre_ids.apply(literal_eval)


# In[ ]:


genres=[]
all_ids=[]
for i in range(len(df)):
    genre_ids=df.genre_ids[i]
    genres.append(genre_ids)
    all_ids.extend(genre_ids)


# In[ ]:


len(all_ids)


# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()
Y=mlb.fit_transform(genres)


# In[ ]:


Y.shape


# In[ ]:


x1 = df.text_vec[0].flatten()
x2 = df.image_vec[0].flatten()


# In[ ]:


x3 = np.concatenate((x1,x2),axis=0)


# In[ ]:


X = np.vstack((X,X))


# In[ ]:


x3.shape


# In[ ]:


X = []
for i in range(len(df)):
  X.append(df.image_vec[i])


# In[ ]:


b=df.text_vec[0].flatten()


# In[ ]:


len(df)


# In[ ]:


a = df.image_vec[0].flatten()


# In[ ]:


a.shape


# In[ ]:


import numpy as np
a = a.flatten()
b = b.flatten()


# In[ ]:


X = np.concatenate((a,b),axis=0)


# In[ ]:


len(X)


# In[ ]:


X.shape


# In[ ]:


X = np.array([])


# In[ ]:


import numpy as np
for i in range(10000, 21489):
  x1 = df.text_vec[i].flatten()
  x2 = df.image_vec[i].flatten()
  x3 = np.concatenate((x1,x2),axis=0)
  X = np.vstack((X,x3))
  if i%5000 == 0:
    print("count is ", i)


# In[ ]:


X.shape


# In[ ]:


X_after = np.delete(X, 0, 0)


# In[ ]:


X_after.shape


# In[ ]:


import pickle 
f=open("inputvecs1_2.pckl", "wb")
pickle.dump(X_after, f, protocol=4)
#pickle.dump(d, open("inputvecs1.pckl", 'w'), protocol=4)

