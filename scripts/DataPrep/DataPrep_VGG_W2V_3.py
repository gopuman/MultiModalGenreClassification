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


x1 = df.text_vec[0]
x2 = df.image_vec[0]


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


import numpy as np
a.shape


# In[ ]:


X = np.concatenate((a,b),axis=0)


# In[ ]:


import numpy as np
a = a.flatten()
b = b.flatten()


# In[ ]:


len(X)


# In[ ]:


X.shape


# In[ ]:


import numpy as np

for i in range(21489, len(df)):
  x1 = df.text_vec[i].flatten()
  x2 = df.image_vec[i].flatten()
  x3 = np.concatenate((x1,x2),axis=0)
  X = np.vstack((X,x3))
  if i%5000 == 0:
    print("count is ", i)


# In[ ]:


import pickle 
f=open("inputvecs2_1.pckl", "wb")
pickle.dump(X_after, f, protocol=4)


# In[ ]:


X_real = X


# In[ ]:


X_real.shape


# In[ ]:


X[0]


# In[ ]:


X_after = np.delete(X, 0, 0)


# In[ ]:


X_after.shape

