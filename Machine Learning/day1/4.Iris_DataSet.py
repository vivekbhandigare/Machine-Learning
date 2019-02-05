#!/usr/bin/env python
# coding: utf-8

# In[11]:


from sklearn import datasets
import matplotlib.pyplot as plt


# In[2]:


iris=datasets.load_iris()


# In[3]:


iris


# In[4]:


print(iris.feature_names)


# In[5]:


iris.data.shape


# In[6]:


print(iris.data[:10])


# In[7]:


iris.target_names


# In[12]:


plt.plot(iris.data[:,:1],iris.data[:,1:2],'ro')
plt.show()


# In[14]:


plt.plot(iris.data[:,2:3],iris.data[:,3:4],'g^')
plt.show()


# In[17]:


plt.plot(iris.data[:,:1],iris.data[:,1:2],'ro',iris.data[:,2:3],iris.data[:,3:4],'g^')
plt.show()


# In[ ]:




