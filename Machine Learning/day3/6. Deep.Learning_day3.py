#!/usr/bin/env python
# coding: utf-8

# # MNIST

# In[1]:


#MODIFIED NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


# In[3]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[4]:


x_train.shape


# In[5]:


x_train[0]


# In[6]:


x_train[0].shape


# In[7]:


plt.imshow(x_train[5500],cmap='gray',interpolation='none')


# In[8]:


x_test.shape


# In[9]:


plt.imshow(x_test[5500],cmap='gray',interpolation='none')


# In[10]:


y_train[5500]


# In[11]:


y_test[5500]


# In[12]:


import vis
vis.imshow_sprite(x_train[:500])


# In[13]:


28*28


# 

# In[14]:


print("x_train shape",x_train.shape)
print("y_train shape",y_train.shape)
print("x_test shape",x_test.shape)
print("y_test shape",y_test.shape)


# In[15]:


x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')


# # Normalize

# In[16]:


x_train/=255
x_test/=255


# # Print shape of final data

# In[17]:


print("train matrix shape",x_train.shape)
print("test matrix shape",x_test.shape)


# In[18]:


np.unique(y_train,return_counts=True)


# # import utils

# In[19]:


from keras.utils import np_utils


# # one-hot encoding

# In[20]:


n_classes=10


# In[21]:


y_orig = y_test
print("Shape before one-hot encoding",y_train.shape)
y_train=np_utils.to_categorical(y_train,n_classes)
y_test=np_utils.to_categorical(y_test,n_classes)
print("Shape after one-hot encoding",y_train.shape)


# In[22]:


y_train[0] #5


# In[23]:


y_train[5500] #1


# # keras neural network imports

# In[24]:


from keras.models import Sequential
from keras.layers.core import Dense,Activation


# In[25]:


model= Sequential()
model.add(Dense(512,input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))


# In[26]:


model.summary()


# # Compile

# In[27]:


model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


# # train

# In[28]:


history= model.fit(x_train,y_train,batch_size=10,epochs=5,validation_data=[x_test,y_test])


# # Lets predict

# In[29]:


y_pred= model.predict_classes(x_test)


# In[30]:


y_pred.shape


# In[31]:


x_temp= x_test[5500].reshape(28,28)


# In[32]:


plt.imshow(x_temp)


# In[42]:


y_pred[5500]


# In[ ]:


#to which tested inputs were detected wrong


# In[43]:


i_i = np.nonzero(y_pred != y_orig)[0]


# In[44]:


len(i_i)


# In[45]:


i_i


# In[46]:


y_pred[115], y_test[115]


# In[ ]:




