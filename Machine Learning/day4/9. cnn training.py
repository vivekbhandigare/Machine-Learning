#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


from keras.datasets import fashion_mnist


# In[8]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[9]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[50]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import vis


# In[24]:


vis.fashion_mnist_label()


# In[25]:


vis.imshow(x_train[0])


# In[29]:


number = 125
plt.imshow(x_train[number],cmap="gray"), print(y_train[number])


# In[30]:


y_train[100]


# In[31]:


x_train_conv = x_train.reshape(x_train.shape[0],28,28,1)
x_test_conv = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train_conv.shape, x_test_conv.shape


# # categorical data

# In[32]:


from keras.utils import to_categorical


# In[33]:


y_train_class = to_categorical(y_train,10)
y_test_class = to_categorical(y_test, 10)


# In[34]:


y_train_class.shape, y_test_class.shape


# # CNN Model

# In[35]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout


# In[36]:


cnn = Sequential()


# In[37]:


cnn.add(Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)))


# In[38]:


cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64, kernel_size=(3,3), activation="relu"))


# In[39]:


cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Flatten())


# In[40]:


cnn.add(Dense(128, activation="relu"))


# In[41]:


cnn.add(Dropout(0.25))
cnn.add(Dense(10, activation="softmax"))


# In[42]:


cnn.summary()


# In[43]:


cnn.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])


# In[44]:


output_cnn = cnn.fit(x_train_conv, y_train_class, batch_size=128, epochs=10, validation_data=(x_test_conv, y_test_class))


# # lets predict

# In[46]:


y_pred= cnn.predict_classes(x_test_conv)


# In[47]:


y_pred.shape


# In[55]:


x_temp= x_test[2]


# In[56]:


plt.imshow(x_temp)


# In[60]:


y_pred[2]


# In[59]:


i_i = np.nonzero(y_pred != y_orig)[0]


# In[ ]:




