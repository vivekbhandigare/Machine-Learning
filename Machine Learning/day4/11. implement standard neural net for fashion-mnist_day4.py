#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


# In[9]:


(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()


# In[ ]:





# In[10]:


X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


# In[11]:


X_train /= 255
X_test /=255


# In[12]:


from keras.utils import np_utils


# In[13]:


n_classes = 10


# In[14]:


y_orig = y_test
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)


# In[24]:


from keras.models import Sequential
from keras.layers.core import Dense, Activation


model= Sequential()
model.add(Dense(512,input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(50, activation="relu"))

model.add(Dense(10))
model.add(Activation('softmax'))


# In[25]:


model.summary()


# In[27]:


model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


# In[28]:


history = model.fit(X_train,y_train,batch_size=10,epochs=5,validation_data=[X_test,y_test])


# In[30]:


y_pred= model.predict_classes(X_test)


# In[31]:


#to which tested inputs were detected wrong
i_i = np.nonzero(y_pred != y_orig)[0]


# In[32]:


len(i_i)


# In[36]:


i_i


# In[ ]:




