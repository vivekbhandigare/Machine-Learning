#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


iris=datasets.load_iris()


# In[3]:


iris


# In[4]:


x = iris.data[:,[2,3]]


# In[5]:


len(x)


# In[6]:


x.shape


# In[7]:


x


# In[8]:


y=iris.target


# In[9]:


y.shape


# In[10]:


y


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


train_test_split


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state =1, stratify = y)


# In[14]:


x.shape


# In[15]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[16]:



from sklearn.preprocessing import StandardScaler        #standardization


# In[17]:


sc = StandardScaler()


# In[18]:


sc.fit(x_train)


# In[19]:


x_train_std =sc.transform(x_train)
x_test_std =sc.transform(x_test)


# In[20]:


sc.transform([[1,2],[3,4]])


# In[21]:


x_train_std


# In[22]:


x_train_std.shape


# # PERCEPTRON
# 
# 

# In[23]:


from sklearn.linear_model import Perceptron


# In[51]:


ppn= Perceptron(n_iter=1000,eta0=0.1,random_state=1)


# In[52]:


ppn


# In[53]:


ppn.fit(x_train_std, y_train)


# # lets predict the output on test data
# 
# 

# In[54]:


y_pred = ppn.predict(x_test_std)


# In[55]:


(y_test != y_pred).sum()


# In[56]:


x_combined_std=np.vstack((x_train_std, x_test_std))
y_combined=np.hstack((y_train, y_test))


# In[57]:


import vis 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier= ppn, test_idx=range(105,150))


# # LOGISTIC REGRESSION ON SAME DATA
# 

# In[59]:


from sklearn.linear_model import LogisticRegression


# In[88]:


lr = LogisticRegression(C = 100.0, random_state=1)


# In[84]:


lr.fit(x_train_std, y_train)


# # predict

# In[85]:


lr.fit(x_train_std, y_train)


# In[86]:


(y_test != y_pred).sum()


# In[87]:


vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier= lr, test_idx=range(105,150))


# # Svm for same data

# In[89]:


from sklearn.svm import SVC


# In[133]:


svm = SVC(kernel='linear', C=1.0,random_state=1)


# In[134]:


svm.fit(x_train_std, y_train)


# # predict

# In[135]:


(y_test != y_pred).sum()


# In[136]:


vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier= svm, test_idx=range(105,150))


# # Decision tree learning on same data

# In[138]:


from sklearn.tree import DecisionTreeClassifier


# In[139]:


tree= DecisionTreeClassifier(criterion='gini',max_depth=4,random_state=1)


# In[140]:


tree.fit(x_train_std, y_train)


# # predict

# In[152]:



(y_test != y_pred).sum()


# In[153]:


vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier= tree, test_idx=range(105,150))   #standardization is not done here


# In[146]:


x_combined=np.vstack((x_train, x_test))
y_combined=np.hstack((y_train, y_test))
vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier= tree, test_idx=range(105,150))


# # Random Forest

# In[158]:


from sklearn.ensemble import RandomForestClassifier


# In[161]:


forest = RandomForestClassifier(criterion='gini',n_estimators=25,random_state=1)


# In[162]:


forest.fit(x_train_std, y_train)


# # predict

# In[169]:


y_pred = forest.predict(x_test_std)
(y_test != y_pred).sum()


# In[170]:


vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier= forest, test_idx=range(105,150))


# # KNN Classfier

# In[174]:


from sklearn.neighbors import KNeighborsClassifier


# In[218]:


knn = KNeighborsClassifier(p=2,n_neighbors=3,metric='minkowski')


# In[219]:


knn.fit(x_train_std, y_train)


# In[220]:


y_pred = knn.predict(x_test_std)
(y_test != y_pred).sum()


# In[221]:


vis.plot_decision_regions(X=x_combined_std, y=y_combined, classifier= knn, test_idx=range(105,150))


# In[ ]:




