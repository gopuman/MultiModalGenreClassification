#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle

f1=open('/content/drive/My Drive/posters_new_features1.pckl','rb')
list_pickled1=pickle.load(f1)
f1.close()

f2=open('/content/drive/My Drive/posters_new_features2.pckl','rb')
list_pickled2=pickle.load(f2)
f2.close()

f3=open('/content/drive/My Drive/posters_new_features3.pckl','rb')
list_pickled3=pickle.load(f3)
f3.close()


# In[ ]:


del f1
del f2
del f3


# In[ ]:


a = list_pickled1[0]
b = list_pickled1[1]
c = list_pickled1[2]
d = list_pickled1[3]
e = list_pickled1[4]

a.extend(list_pickled2[0])
b.extend(list_pickled2[1])
c.extend(list_pickled2[2])
d.extend(list_pickled2[3])
e.extend(list_pickled2[4])

a.extend(list_pickled3[0])
b.extend(list_pickled3[1])
c.extend(list_pickled3[2])
d.extend(list_pickled3[3])
e.extend(list_pickled3[4])


# In[ ]:


final_feature_list = a
final_files = b
final_failed = c
final_succesful = d
final_genre_list = e


# In[ ]:


del a
del b
del c
del d
del e


# In[6]:


import numpy as np
(feature_list,files,failed,succesful,genre_list)=(final_feature_list,final_files,final_failed,final_succesful,final_genre_list)

(a,b,c,d)=feature_list[0].shape
feature_size=a*b*c*d

np_features=np.zeros((len(feature_list),feature_size))
for i in range(len(feature_list)):
    feat=feature_list[i]
    reshaped_feat=feat.reshape(1,-1)
    np_features[i]=reshaped_feat

X=np_features

from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()
Y=mlb.fit_transform(genre_list)

Y.shape


# In[ ]:


del feat
del feature_list
del feature_size
del final_failed
del final_feature_list
del final_files
del final_genre_list
del final_succesful
del genre_list
del list_pickled1
del list_pickled2
del list_pickled3
del reshaped_feat
del succesful
del np_features
del mlb


# In[ ]:


del failed
del files
del MultiLabelBinarizer


# In[ ]:


mask = np.random.rand(len(X)) < 0.8

X_train=X[mask]
X_test=X[~mask]
Y_train=Y[mask]
Y_test=Y[~mask]


# In[ ]:


del X
del Y


# In[11]:


visual_problem_data=(X,Y)
f8=open('visual_data.pckl','wb')
pickle.dump(visual_problem_data,f8)
f8.close()


# In[12]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
model_visual = Sequential([
    Dense(1024, input_shape=(25088,)),
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


# In[13]:


model_visual.fit(X_train, Y_train, epochs=10, batch_size=64,verbose=1)


# In[14]:


model_visual.fit(X_train, Y_train, epochs=50, batch_size=64,verbose=1)


# In[ ]:


Y_preds=model_visual.predict(X_test)


# In[16]:


sum(sum(Y_preds))


# In[ ]:


f6=open('/content/drive/My Drive/Genredict.pckl','rb')
Genre_ID_to_name=pickle.load(f6)
f6.close()


# In[20]:


sum(Y_preds[1])
sum(Y_preds[2])


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


# In[23]:


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

print(np.mean(np.asarray(precs)),np.mean(np.asarray(recs)))
print("avg=",(np.mean(np.asarray(precs))+np.mean(np.asarray(recs)))/2)


# In[24]:


from keras.models import model_from_json

model_json = model_visual.to_json()
with open("VGG.json", "w") as json_file:
    json_file.write(model_json)
model_visual.save_weights("VGG.h5")
print("Saved model to disk")


# In[ ]:


# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#features = model.predict(x)

img = image.load_img('/content/drive/My Drive/storage3/Iron_Man.jpg',target_size=(224,224))
img = image.img_to_array(img)
img = np.expand_dims(img,axis=0)
img = preprocess_input(img)


# In[ ]:


features = model.predict(img)


# In[ ]:


proba = model_visual.predict(features.reshape(1,-1))


# In[31]:


proba


# In[ ]:


top_3 = np.argsort(proba[0])[:-4:-1]


# In[33]:


top_3


# In[34]:


for genre in top_3:
       print(Genre_ID_to_name[genre_list[genre]])


# In[ ]:




