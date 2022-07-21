#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle

f1=open('/content/drive/My Drive/posters_new_features1.pckl','rb')
#f2=open('/content/drive/My Drive/posters_new_features2.pckl','rb')
#f3=open('/content/drive/My Drive/posters_new_features3.pckl','rb')


list_pickled1=pickle.load(f1)
#list_pickled2=pickle.load(f2)
#list_pickled3=pickle.load(f3)

f1.close()
#f2.close()
#f3.close()


# In[ ]:


list_pickled1[4]


# In[ ]:


del list_pickled1
del list_pickled2
del list_pickled3


# In[ ]:


del f1
del f2
del f3


# In[ ]:


type(list_pickled1)


# In[ ]:


def find_length(p):
  for i in p:
    print(len(i))


# In[ ]:


find_length(list_pickled1)


# 39133 Results

# In[ ]:


final_feature_list = list_pickled1[0] + list_pickled2[0] + list_pickled3[0] 
final_files = list_pickled1[1] + list_pickled2[1] + list_pickled3[1]
final_failed = list_pickled1[2] + list_pickled2[2] + list_pickled3[2]
final_succesful = list_pickled1[3] + list_pickled2[3] + list_pickled3[3]
final_genre_list = list_pickled1[4] + list_pickled2[4] + list_pickled3[4]


# In[ ]:


del final_files
del final_failed
del final_succesful
del final_genre_list


# In[ ]:


import pickle
list_pickled=(final_feature_list,final_files,final_failed,final_succesful,final_genre_list)
f=open('posters_final.pckl','wb')
pickle.dump(list_pickled,f)
f.close()


# In[ ]:


f7=open('posters_new_features.pckl','rb')
list_pickled=pickle.load(f7)
f7.close()


# In[ ]:


import numpy as np

(a,b,c,d)=final_feature_list[0].shape
feature_size=a*b*c*d

np_features=np.zeros((len(final_feature_list),feature_size))
for i in range(len(final_feature_list)):
    feat=final_feature_list[i]
    reshaped_feat=feat.reshape(1,-1)
    np_features[i]=reshaped_feat

X=np_features

from sklearn.preprocessing import MultiLabelBinarizer
mlb=MultiLabelBinarizer()
Y=mlb.fit_transform(final_genre_list)

Y.shape


# In[ ]:


visual_problem_data=(X,Y)
#f8=open('visual_problem_data_clean.pckl','wb')
#pickle.dump(visual_problem_data,f8)
#f8.close()


# In[ ]:


#import pickle
#f8=open('visual_problem_data_clean.pckl','rb')
visual_features=visual_problem_data
#f8.close()


# In[ ]:


(X,Y)=visual_features


# In[ ]:


X.shape


# In[ ]:


mask = np.random.rand(len(X)) < 0.8


# In[ ]:


X_train=X[mask]
X_test=X[~mask]
Y_train=Y[mask]
Y_test=Y[~mask]


# In[ ]:


X_test.shape


# In[ ]:


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


# In[ ]:


model_visual.fit(X_train, Y_train, epochs=10, batch_size=64,verbose=1)


# In[ ]:


model_visual.fit(X_train, Y_train, epochs=50, batch_size=64,verbose=1)


# In[ ]:


Y_preds=model_visual.predict(X_test)


# In[ ]:


sum(sum(Y_preds))


# In[ ]:


f6=open('/content/drive/My Drive/Genredict.pckl','rb')
Genre_ID_to_name=pickle.load(f6)
f6.close()


# In[ ]:


sum(Y_preds[1])


# In[ ]:


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


# In[ ]:


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




