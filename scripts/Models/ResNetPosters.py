#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle

f1 = open("/content/drive/My Drive/Resnet_features_1.pckl","rb")
x1 = pickle.load(f1)


# In[ ]:


f2 = open("/content/drive/My Drive/Resnet_features_2.pckl","rb")
x2 = pickle.load(f2)


# In[ ]:


f3 = open("/content/drive/My Drive/Resnet_features_3.pckl","rb")
x3 = pickle.load(f3)


# In[ ]:


X = x1[0]
X.extend(x2[0])
X.extend(x3[0])


# In[ ]:


genre_list = x1[4]
genre_list.extend(x2[4])
genre_list.extend(x3[4])


# In[31]:


feature_size=1000

np_features=np.zeros((len(X),feature_size))
for i in range(len(X)):
    feat=X[i]
    reshaped_feat=feat.reshape(1,-1)
    np_features[i]=reshaped_feat

X=np_features

from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()
Y=mlb.fit_transform(genre_list)

Y.shape


# In[ ]:


mask = np.random.rand(len(X)) < 0.8


# In[ ]:


X_train=X[mask]
X_test=X[~mask]
Y_train=Y[mask]
Y_test=Y[~mask]


# In[34]:


X_test.shape


# In[35]:


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
model_visual = Sequential([
    Dense(1024, input_shape=(1000,)),
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


# In[36]:


model_visual.fit(X_train, Y_train, epochs=60, batch_size=64,verbose=1)


# In[ ]:


Y_preds=model_visual.predict(X_test)


# In[38]:


sum(sum(Y_preds))


# In[ ]:


f6=open('/content/drive/My Drive/Genredict.pckl','rb')
Genre_ID_to_name=pickle.load(f6)
f6.close()


# In[40]:


sum(Y_preds[1])


# In[ ]:


genre_list=sorted(list(Genre_ID_to_name.keys()))


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


# In[43]:


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


# In[44]:


print(np.mean(np.asarray(precs)),np.mean(np.asarray(recs)))
print("avg=",(np.mean(np.asarray(precs))+np.mean(np.asarray(recs)))/2)


# In[45]:


print("f1-score", (2*(np.mean(np.asarray(precs)) * np.mean(np.asarray(recs)))) / (np.mean(np.asarray(precs)) + np.mean(np.asarray(recs))))

