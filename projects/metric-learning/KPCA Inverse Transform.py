
# coding: utf-8

# In[1]:

from sklearn.decomposition import KernelPCA


# In[2]:

import matplotlib.pyplot as plt


# In[3]:

get_ipython().magic(u'matplotlib inline')


# In[4]:

from sklearn.datasets import load_digits
X,Y = load_digits().data,load_digits().target


# In[7]:

kpca = KernelPCA(gamma=0.1,kernel='rbf',fit_inverse_transform=True)
X_ = kpca.fit_transform(X)


# In[8]:

plt.scatter(X_[:,0],X_[:,1],c=Y,cmap='rainbow',linewidths=0,s=4)


# In[ ]:



