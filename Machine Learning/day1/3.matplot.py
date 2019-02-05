#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[5]:


plt.plot([1,2,3,4])
plt.ylabel('Y-label')
plt.xlabel('x-label')
plt.show()


# In[12]:


plt.plot([1,2,3,4],[1,3,3,4],'gs')
plt.axis([0,5,0,5])
plt.show()


# In[14]:


import numpy as np
t= np.arange(0.,5.,0.2)

plt.plot(t,t,'r--',t**2,'bs',t,t**3,'g^')
plt.axis([0,5,0,5])
plt.show()


# In[15]:


import imageio
from skimage import transform
im=imageio.imread('imageio:chelsea.png')
plt.imshow(im)


# In[16]:


im


# In[18]:


im.shape


# In[19]:


lum_img=im[:,:,0]
plt.imshow(lum_img)


# In[22]:


lum_img=im[:,:,2]
plt.imshow(lum_img)


# In[35]:


lum_img=im[:,:,0]
plt.imshow(lum_img,cmap='hot')


# In[37]:


im=imageio.imread('imageio:chelsea.png')
im_next=transform.resize(im,(64,64))
implot=plt.imshow(im_next)


# In[38]:





# In[ ]:




