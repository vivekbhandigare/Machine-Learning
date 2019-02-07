#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import imdb


# In[2]:


from keras import preprocessing


# In[3]:


max_features=10000


# In[4]:


maxlen = 20


# In[5]:


(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)


# In[8]:


x_train.shape,x_test.shape, y_train.shape,y_test.shape


# In[9]:


len(x_train[0]),len(x_train[1])


# In[10]:


y_train[0],y_train[1]


# In[11]:


import numpy as np 
np.unique(y_train)


# In[12]:


x_train=preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen)
x_test=preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)


# In[13]:


x_train.shape,x_test.shape


# In[14]:


x_train[0]


# # Model

# In[15]:


from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding


# 

# In[17]:


model = Sequential()
model.add(Embedding(10000,8,input_length=maxlen))


# In[18]:


model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))


# In[19]:


model.summary()


# # Learn

# In[27]:


model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['acc'])


# # train

# In[28]:


history=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2)


# # looking at model history

# In[33]:


acc=history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[35]:


import matplotlib.pyplot as plt
epochs = range(1,len(acc)+1)


# In[36]:


plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc,'b', label = 'Validation acc')
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()


# In[ ]:




