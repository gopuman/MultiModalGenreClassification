#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_pickle("/content/drive/My Drive/pinky_promise_last.pckl")


# In[ ]:


df


# In[ ]:


df.text_vec[0].shape


# In[ ]:


l = df.image_vec.to_list()


# In[ ]:


import numpy as np
import pickle
X = np.array([])


# In[ ]:


f=open('/content/drive/My Drive/glove_features.pckl','rb')
glove = pickle.load(f)


# In[ ]:


g = []
for i in range(len(glove[0])):
  g.append(glove[0][i])


# In[ ]:


g[0].shape


# In[ ]:


df.image_vec[0].flatten().shape


# In[ ]:


a = df.image_vec[0].flatten()
b = g[0].flatten()
X = np.concatenate((a,b),axis=0)


# In[ ]:


X.shape


# In[ ]:


import numpy as np
for i in range(21489, len(df)):
  x1 = g[i].flatten()
  x2 = df.image_vec[i].flatten()
  x3 = np.concatenate((x1,x2),axis=0)
  X = np.vstack((X,x3))
  if i%5000 == 0:
    print("count is ", i)


# In[ ]:


len(X)


# In[ ]:


X_after = np.delete(X, 0, 0)


# In[ ]:


X_after[0]


# In[ ]:


import pickle 
f=open("glovecombivecs2.pckl", "wb")
pickle.dump(X_after, f, protocol=4)
#pickle.dump(d, open("inputvecs1.pckl", 'w'), protocol=4)

