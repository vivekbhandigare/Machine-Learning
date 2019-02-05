#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
list = [[1,2,3],[4,5,6]]
arr2d = np.array(list)
arr2d


# In[4]:


arr2d.size


# 

# In[5]:


arr2d.shape  


# In[6]:


arr2d.ndim


# In[7]:


arr2d.itemsize


# In[8]:


arr2d.dtype


# In[9]:


arr_float = arr2d.astype(np.float32)
arr_float


# In[13]:


arr_fourbyfive = _1st = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]
arr_fourbyfive = np.array(arr_fourbyfive,dtype='float32')
arr_fourbyfive


# In[17]:


np.zeros((3,3))


# In[19]:


np.eye(4)


# In[20]:


np.diag((2,2))


# In[21]:


np.random.rand(3,4)


# In[22]:


np.arange(6)


# In[26]:


np.arange(7,10)


# In[27]:


np.arange(1,9,2)


# In[29]:


np.linspace(0.,1.,num=5)


# In[33]:


np.arange(1,40,2)    #odd no. between 0 to 40


# In[36]:


arr_ind=np.array([1,2,3])
arr_ind


# In[37]:


arr2d[:2]


# In[38]:


arr2d[0, :2]


# In[39]:


arr_fourbyfive    #it was defined earlier


# In[40]:


arr_fourbyfive[2, :]      #access row


# In[41]:


arr_fourbyfive[:,2]       #access a column


# In[44]:


arr_fourbyfive[:,:3]     #access first 3 column


# In[46]:


arr_fourbyfive[:2, :]        #access first 2 rows


# In[52]:


arr_fourbyfive[:3, :4]  #access first 3 rows and 4 column


# In[65]:


arr_fourbyfive[::-1,::-1]      #reverse


# In[75]:


arr_fourbyfive[:,-1] 


# In[76]:


arr2d=np.array([[1,2,3],[4,5,6]])
print(np.add(arr2d,4))
arr2d


# In[78]:


print(np.subtract(arr2d,2))


# In[79]:


print(arr_fourbyfive*4)


# In[80]:


arr2d **2


# In[81]:


new_arr = np.array([[1,2,3],[4,5,6]])
np.add.reduce(new_arr)
                


# In[82]:


np.add.reduce(new_arr, axis=1)


# In[86]:


arr3d = np.array([1,2,4])+1
arr3d


# In[91]:


#reshaping an array

arr1d=np.array([4,5,6,8,7,45,89,6,7,6,8,9])
arr4d_view=arr1d.reshape(3,4)
arr4d_view


# In[94]:


arr4d_view.ravel()


# In[96]:





# In[100]:





# In[ ]:




