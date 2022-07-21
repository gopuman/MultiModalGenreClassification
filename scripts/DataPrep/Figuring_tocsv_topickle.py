#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


df1 = pd.read_csv("/content/drive/My Drive/s1_imagevec.csv")
df2 = pd.read_csv("/content/drive/My Drive/s2_imagevec.csv")
df3 = pd.read_csv("/content/drive/My Drive/s3_imagevec.csv")


# In[ ]:


len(df1) + len(df2) + len(df3)


# In[ ]:


df = pd.concat([df1,df2,df3])


# In[ ]:


len(df)


# In[ ]:


df = df.reset_index()


# In[ ]:


df = df.drop("index",axis=1)
df = df.drop("Unnamed: 0",axis=1)


# In[ ]:


x1 = df.text_vec[0]


# In[ ]:


x1 = x1.split(" ")


# In[ ]:


X1=[]
x1 = df.text_vec[0].split(" ")
for j in x1:
  if(j):
    x1.pop(0)
    j = j.rstrip("]")
    X1.append(j)
X.append(np.asarray(X1,dtype=float))


# In[ ]:


df


# In[ ]:


X=[]


# In[ ]:


#Text_vec restructuring

for i in range(len(df)):
  X1=[]
  x1 = df.text_vec[i].split(" ")
  for j in x1:
    j = j.rstrip("]")
    j = j.lstrip("[")
    j = j.rstrip("\n")
    if(j):
      X1.append(float(j))
  X.append(X1)

# for j in x1:
#   if(j):
#     x1.pop(0)
#     j = j.rstrip("]")
#     j = j.rstrip("\n")
#     x3.append(j)


# In[ ]:


x2 = df.image_vec[0]
x2


# In[ ]:


import pickle


# In[ ]:


f1 = open("/content/drive/My Drive/posters_new_features1.pckl",'rb')
f2 = open("/content/drive/My Drive/posters_new_features2.pckl",'rb')
f3 = open("/content/drive/My Drive/posters_new_features3.pckl",'rb')


# In[ ]:


feat1 = pickle.load(f1)
feat2 = pickle.load(f2)
feat3 = pickle.load(f3)


# In[ ]:


len(feat1[0][3])


# In[ ]:


df = df.drop("image_vec",axis=1)


# In[ ]:


df


# In[ ]:


df3 = df3.reset_index()
df3 = df3.drop("index",axis=1)
df3 = df3.drop("Unnamed: 0",axis=1)


# In[ ]:


df3


# In[ ]:


df3 = df3.drop("image_vec",axis=1)


# In[ ]:


movs = feat3[3]
movie = []


# In[ ]:


movs = feat3[3]
movie = []
for i in movs:
  i = i.replace("_"," ")
  i = i.rstrip(".jpg")
  movie.append(i)


# In[ ]:


movie


# In[ ]:


final_feats_3 = []
for i in range(len(df3)):
  try:
    ind = movie.index(df3.title[0])
    final_feats_3.append(feat3[0][ind])
  except:
    final_feats_3.append("NaN")


# In[ ]:


len(final_feats_3)


# In[ ]:


df3["image_vec"] = final_feats_3


# In[ ]:


df3


# In[ ]:


df2.image_vec[0]


# In[ ]:


df = pd.DataFrame()


# In[ ]:


df = pd.concat([df1,df2,df3])


# In[ ]:


len(df)


# In[ ]:


df = df.reset_index()


# In[ ]:


df=df.drop("index",axis=1)


# In[ ]:


df


# In[ ]:


df.to_csv("promise_last.csv")


# In[ ]:


df.image_vec[0]


# In[ ]:


df1 = pd.read_csv("promise_last.csv",dtype={"image_vec":"float32"})


# In[ ]:


type(df1.image_vec[0])


# In[ ]:


df.to_pickle("promise_last.pckl")


# In[ ]:


dfx = pd.read_pickle("promise_last.pckl")


# In[ ]:


type(dfx.image_vec[0])

