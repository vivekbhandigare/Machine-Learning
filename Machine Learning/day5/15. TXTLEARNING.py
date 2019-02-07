#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


# In[2]:


model=ResNet50(weights='imagenet')


# In[28]:


img_path='mug.jpg'
img=image.load_img(img_path,target_size=(224,224))


# In[29]:


import numpy as np
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)


# In[30]:


x.shape


# In[31]:


preds=model.predict(x)


# In[32]:


print('Predict:',decode_predictions(preds, top=3)[0])


# In[ ]:




