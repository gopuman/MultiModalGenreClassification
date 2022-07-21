#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install beautifulsoup4')


# In[ ]:


from bs4 import BeautifulSoup
import requests
import urllib.request
import argparse
import os


# In[13]:


page = "http://www.readcomiconline.to/Comic/The-Amazing-Spider-Man-1963/Annual-1?id=34815&readType=1"
page1 = "www.google.com"
result = requests.get(page)

if result.status_code == 200:
  soup = BeautifulSoup(result.content,"html.parser")
else:
  print(result.status_code)

#div = soup.find('div',{'id':'divImage'})



# In[ ]:




