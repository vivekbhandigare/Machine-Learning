#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.preprocessing.text import Tokenizer


# In[35]:


sample = ["The Cat sat on the mat","The Dog ate my homework"]


# In[42]:


tokenizer= Tokenizer(num_words=10)


# In[43]:


tokenizer.fit_on_texts(sample)


# In[44]:


sequences = tokenizer.texts_to_sequences(sample)


# In[45]:


sequences


# # one hot encoding
# 

# In[46]:


one_hot_results = tokenizer.texts_to_matrix(sample,mode='binary')


# In[47]:


one_hot_results


# In[48]:


counts = tokenizer.texts_to_matrix(sample,mode='count')


# In[49]:


counts


# In[50]:


one_hot_tfidf = tokenizer.texts_to_matrix(sample,mode='tfidf')


# In[52]:


one_hot_tfidf


# In[53]:


word_index = tokenizer.word_index


# In[54]:


word_index


# In[ ]:




