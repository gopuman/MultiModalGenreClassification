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


len(X_after)


# In[ ]:


X_after.shape


# In[ ]:


import pickle 
f=open("glovecombivecs2.pckl", "wb")
pickle.dump(X_after, f, protocol=4)
#pickle.dump(d, open("inputvecs1.pckl", 'w'), protocol=4)


# In[ ]:


del X
del X_after


# In[ ]:


import pickle

f1 = open("/content/drive/My Drive/glovecombivecs1.pckl","rb")
x1 = pickle.load(f1)


# In[ ]:


f2 = open("/content/drive/My Drive/glovecombivecs2.pckl","rb")
x2 = pickle.load(f2)


# In[ ]:


del f1
del f2


# In[ ]:


import numpy as np
X = np.concatenate((x1,x2),axis=0)


# In[ ]:


X.shape


# In[ ]:


del x1 
del x2


# In[ ]:


import pandas as pd
import pickle
df = pd.read_pickle("/content/drive/My Drive/pinky_promise_last.pckl")


# In[ ]:


Y = df.genre_ids.to_list()


# In[ ]:


Y


# In[ ]:


mask = np.random.rand(len(X)) < 0.8


# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()
Y=mlb.fit_transform(Y)

Y.shape


# In[ ]:


X_train=X[mask]
X_test=X[~mask]
Y_train=Y[mask]
Y_test=Y[~mask]


# In[ ]:


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


# In[ ]:


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


# In[ ]:


model_visual.fit(X_train, Y_train, epochs=150, batch_size=64,verbose=1)


# In[ ]:


model_visual.fit(X_train, Y_train, epochs=50, batch_size=64,verbose=1)


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


print(np.mean(np.asarray(precs)),np.mean(np.asarray(recs)))
print("avg=",(np.mean(np.asarray(precs))+np.mean(np.asarray(recs)))/2)


# In[ ]:


print("f1-score", (2*(np.mean(np.asarray(precs)) * np.mean(np.asarray(recs)))) / (np.mean(np.asarray(precs)) + np.mean(np.asarray(recs))))


# In[1]:


get_ipython().system('cat /proc/cpuinfo')


# In[ ]:




