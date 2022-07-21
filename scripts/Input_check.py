#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import model_from_json

json_file = open('/content/drive/My Drive/Resnet_Glove.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/drive/My Drive/Resnet_Glove.h5")
print("Loaded model from disk")


# In[ ]:




