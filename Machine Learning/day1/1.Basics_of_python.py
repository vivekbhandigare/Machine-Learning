#!/usr/bin/env python
# coding: utf-8

# In[1]:


greeting = "hello world"
greeting[0]='j'
greeting


# In[22]:


le_str='dhgh dhg gd hgdj'
words=le_str.split()
len(words)
words


# In[4]:


for words in words:
    print(words)


# In[7]:


le_str.find('hg')


# In[36]:


a=le_str[-1:-4:-1]


# In[37]:


a


# In[39]:


print(le_str.upper())


# In[40]:


nested_list= [[1,2],[2,4],[5,66,6]]
nested_list[2]


# In[41]:


nested_list[0][1]


# In[42]:


nested_list[:1]


# In[43]:


nested_list[::-1]


# In[49]:


for i in nested_list:
    for j in i:
        print(j)


# In[54]:


for h in range(1,10):
    print(h)


# In[5]:


for h in range(1,10):
    for k in h:
        print(h)
    


# In[19]:


ab={'k1':'we','k2':'do'}
ab['k1']


# In[22]:


ab['k3']=['jk','gh','gj']


# In[23]:


ab


# In[28]:


if 'k3' in ab:
    print(ab['k2'])
else:
    print('No Key')


# In[29]:


ab.items()


# In[30]:


ab.keys()


# In[35]:


ab['k3'].append('kit')
ab['k3']


# In[ ]:


#tuples

