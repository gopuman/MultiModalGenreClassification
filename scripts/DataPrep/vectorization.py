#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()
print(c)

print("Vectorized version "+str(1000*(toc-tic))+"ms")


# In[5]:


c = 0
tic = time.time()
for i in range(len(a)):
  c+=a[i]*b[i]
toc = time.time()
print(c)

print("Non-Vectorized version "+str(1000*(toc-tic))+"ms")


# In[2]:


import numpy as np

a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a*b


# In[5]:


c


# In[ ]:




