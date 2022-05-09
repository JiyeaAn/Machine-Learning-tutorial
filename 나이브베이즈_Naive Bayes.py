#!/usr/bin/env python
# coding: utf-8

# ## Gaussian Naive Bayes

# In[1]:


from sklearn import datasets
from sklearn.naive_bayes import GaussianNB


# In[2]:


import pandas as pd


# In[3]:


iris = datasets.load_iris()
df_X=pd.DataFrame(iris.data)
df_Y=pd.DataFrame(iris.target)


# In[4]:


df_X.head()


# In[5]:


df_Y.head()


# In[6]:


gnb=GaussianNB()
fitted=gnb.fit(iris.data,iris.target)
y_pred=fitted.predict(iris.data)


# In[7]:


fitted.predict_proba(iris.data)[[1,48,51,100]]


# In[8]:


fitted.predict(iris.data)[[1,48,51,100]]


# In[9]:


from sklearn.metrics import confusion_matrix


# In[10]:


confusion_matrix(iris.target,y_pred)


# In[11]:


gnb2=GaussianNB(priors=[1/100,1/100,98/100])
fitted2=gnb2.fit(iris.data,iris.target)
y_pred2=fitted2.predict(iris.data)
confusion_matrix(iris.target,y_pred2)


# In[12]:


gnb2=GaussianNB(priors=[1/100,98/100,1/100])
fitted2=gnb2.fit(iris.data,iris.target)
y_pred2=fitted2.predict(iris.data)
confusion_matrix(iris.target,y_pred2)


# ## Multinomial naive bayes

# In[13]:


from sklearn.naive_bayes import MultinomialNB


# In[14]:


import numpy as np


# In[15]:


X = np.random.randint(5, size=(6, 100))
y = np.array([1, 2, 3, 4, 5, 6])


# In[16]:


X


# In[17]:


y


# In[18]:


clf = MultinomialNB()
clf.fit(X, y)


# In[19]:


print(clf.predict(X[2:3]))


# In[20]:


clf.predict_proba(X[2:3])


# In[21]:


clf2 = MultinomialNB(class_prior=[0.1,0.5,0.1,0.1,0.1,0.1])
clf2.fit(X, y)


# In[22]:


clf2.predict_proba(X[2:3])

