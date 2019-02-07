#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.datasets import imdb
from keras.preprocessing import sequence


# In[3]:


max_features=10000
maxlen = 500


# In[4]:


(input_train,y_train),(input_test,y_test)=imdb.load_data(num_words=max_features)


# In[5]:


input_train


# In[7]:


input_test


# In[8]:


input_train.shape, input_test.shape


# In[11]:


x_train=sequence.pad_sequences(input_train,maxlen=maxlen)
x_test=sequence.pad_sequences(input_test,maxlen=maxlen)


# In[12]:


x_train.shape,x_test.shape


# In[13]:


x_train


# # RNN

# In[15]:


from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding


# In[17]:


model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))


# In[18]:


model.summary()


# In[19]:


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])


# In[22]:


history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)


# # LSTM

# In[23]:


from keras.layers import LSTM


# In[24]:


model_lstm = Sequential()
model_lstm.add(Embedding(max_features, 32))
model_lstm.add(LSTM(32))
model_lstm.add(Dense(1,activation='sigmoid'))


# In[25]:


model_lstm.summary()


# In[26]:


model_lstm.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])


# In[29]:


history = model_lstm.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2 )


# In[ ]:




