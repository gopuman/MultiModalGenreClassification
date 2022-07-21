#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df1 = pd.read_csv("/content/drive/My Drive/s1_imagevec.csv")
df2 = pd.read_csv("/content/drive/My Drive/s2_imagevec.csv")
df3 = pd.read_csv("/content/drive/My Drive/s3_imagevec.csv")


# In[5]:


len(df1) + len(df2) + len(df3)


# In[ ]:


df = pd.concat([df1,df2,df3])


# In[7]:


len(df)


# In[ ]:


df = df.reset_index()


# In[ ]:


df = df.drop("index",axis=1)
#df = df.drop("Unnamed: 0",axis=1)


# In[14]:


df


# In[ ]:


x1 = df.text_vec[0]


# In[16]:


x1


# In[ ]:


import numpy as np


# In[ ]:


mystr = '[1 2 3 4 5]'
x=(mystr)


# In[ ]:


x.remove("]")


# In[86]:


x


# In[ ]:


x=np.array(x)


# In[88]:


x


# In[ ]:


x = str(x).replace("'", "")


# In[90]:


x

