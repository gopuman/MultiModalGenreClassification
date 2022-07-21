#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('ls')


# In[ ]:


import os
poster_movies_2 = os.listdir("/content/drive/My Drive/storage2")
print(len(poster_movies_2))


# In[ ]:


poster_movies = poster_movies_2


# In[ ]:


len(set(poster_movies))


# In[ ]:


poster_movies_copy_2 = [i.replace("_"," ").rstrip(".jpg") for i in poster_movies_2]


# In[ ]:


get_ipython().system('pip install tensorflow==1.14')


# In[ ]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)


# In[ ]:


import pandas as pd

df = pd.read_csv("/content/drive/My Drive/last.csv", engine="python")
df2 = df[df['title'].isin(poster_movies_copy_2)]
df2 = df2.reset_index()


# In[ ]:


len(df2)


# In[ ]:


imnames=poster_movies
feature_list2=[]
genre_list2=[]
file_order2=[]
print("Starting extracting VGG features for scraped images. This will take time, Please be patient...")
print("Total images = ",len(imnames))
failed_files2=[]
succesful_files2=[]
i=0
for mov in range(len(df2)):
    i+=1
    mov_name=df2.title[mov]
    mov_name=mov_name.replace(':','/')
    poster_name=mov_name.replace(' ','_')+'.jpg'
    if poster_name in imnames:
        img_path="/content/drive/My Drive/storage2/"+poster_name
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            succesful_files2.append(poster_name)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = model.predict(x)
            file_order2.append(img_path)
            feature_list2.append(features)
            genre_list2.append(df2.genre_ids[mov])
            if np.max(np.asarray(feature_list2))==0.0:
                print('problematic',i)
            if i%50==0 or i==1:
                print("Working on Image : ",i)
        except:
            failed_files2.append(poster_name)
            continue
        
    else:
        continue
print("Done with all features, please pickle for future use!")


# In[ ]:


len(succesful_files2)


# In[ ]:


len(failed_files2)


# In[ ]:


import pickle
list_pickled=(feature_list2,file_order2,failed_files2,succesful_files2,genre_list2)
f=open('posters_new_features2.pckl','wb')
pickle.dump(list_pickled,f)
f.close()
print("Features dumped to pickle file")


# In[ ]:


import pickle
f7=open('/content/drive/My Drive/posters_new_features2.pckl','rb')
list_pickled=pickle.load(f7)
f7.close()


# In[ ]:


#Pickle is indexed, can get individual info out of them
list_pickled[2]


# In[ ]:


import numpy as np
(feature_list,files,failed,succesful,genre_list)=list_pickled

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


visual_problem_data=(X,Y)
f8=open('visual_problem_data_clean.pckl','wb')
pickle.dump(visual_problem_data,f8)
f8.close()


# In[ ]:


f8=open('visual_problem_data_clean.pckl','rb')
visual_features=pickle.load(f8)
f8.close()


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




