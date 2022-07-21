#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle

f1 = open("/content/drive/My Drive/inputvecs1_1.pckl",'rb')
x1 = pickle.load(f1)


# In[ ]:


f2 = open("/content/drive/My Drive/inputvecs1_2.pckl","rb")
x2 = pickle.load(f2)


# In[4]:


x1.shape


# In[ ]:


import numpy as np
X = np.concatenate((x1,x2),axis=0)


# In[6]:


X.shape


# In[ ]:


f3 = open("/content/drive/My Drive/inputvecs2.pckl","rb")
x3 = pickle.load(f3)


# In[ ]:


del f1
del f2
del x1 
del x2
del f3


# In[ ]:


X = np.concatenate((X,x3),axis=0)


# In[ ]:


del x3


# In[11]:


X.shape


# In[ ]:


import pandas as pd
import pickle
df = pd.read_pickle("/content/drive/My Drive/pinky_promise_last.pckl")


# In[ ]:


Y = df.genre_ids.to_list()


# In[14]:


Y


# In[ ]:


import numpy as np
mask = np.random.rand(len(X)) < 0.8


# In[16]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()
Y=mlb.fit_transform(Y)

Y.shape


# In[ ]:


X_train=X[mask]
X_test=X[~mask]
Y_train=Y[mask]
Y_test=Y[~mask]


# In[18]:


dir()


# In[ ]:


del df
del mlb
del Y
del pickle
del np 
del pd
del X
del MultiLabelBinarizer


# In[20]:


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
model_visual = Sequential([
    Dense(1024, input_shape=(25388,)),
    Activation('relu'),
    Dense(256),
    Activation('relu'),
    Dense(14),
    Activation('sigmoid'),
])
opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

#sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.4, nesterov=False)
model_visual.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[21]:


model_visual.fit(X_train, Y_train, epochs=200, batch_size=64,verbose=1)


# In[ ]:


def precision_recall(gt,preds):
    TP=0
    FP=0
    FN=0
    for t in gt:
        if t in preds:
            TP+=1
        else:
            FN+=1
    for p in preds:
        if p not in gt:
            FP+=1
    if TP+FP==0:
        precision=0
    else:
        precision=TP/float(TP+FP)
    if TP+FN==0:
        recall=0
    else:
        recall=TP/float(TP+FN)
    return precision,recall


# In[ ]:


Y_preds=model_visual.predict(X_test)


# In[ ]:


import pickle
f6=open('/content/drive/My Drive/Genredict.pckl','rb')
Genre_ID_to_name=pickle.load(f6)
f6.close()


# In[ ]:


genre_list=sorted(list(Genre_ID_to_name.keys()))


# In[ ]:


del f6 
del pickle


# In[36]:


sum(sum(Y_preds))


# In[ ]:


import pandas as pd
df = pd.read_pickle("/content/drive/My Drive/pinky_promise_last.pckl")


# In[ ]:


import numpy as np
positions=range(len(df))
# print positions
test_movies=np.asarray(positions)[~mask]
# test_movies


# In[40]:


import numpy as np

precs=[]
recs=[]
for i in range(len(Y_preds)):
    row=Y_preds[i]
    gt_genres=Y_test[i]
    gt_genre_names=[]
    for j in range(14):
        if gt_genres[j]==1:
            gt_genre_names.append(Genre_ID_to_name[genre_list[j]])
    top_3=np.argsort(row)[-4:]
    predicted_genres=[]
    for genre in top_3:
        predicted_genres.append(Genre_ID_to_name[genre_list[genre]])
    (precision,recall)=precision_recall(gt_genre_names,predicted_genres)
    precs.append(precision)
    recs.append(recall)
    print("Predicted: ",','.join(predicted_genres)," Actual: ",','.join(gt_genre_names))


# In[41]:


print(np.mean(np.asarray(precs)),np.mean(np.asarray(recs)))
print("avg=",(np.mean(np.asarray(precs))+np.mean(np.asarray(recs)))/2)


# In[42]:


print("f1-score", (2*(np.mean(np.asarray(precs)) * np.mean(np.asarray(recs)))) / (np.mean(np.asarray(precs)) + np.mean(np.asarray(recs))))


# In[ ]:




