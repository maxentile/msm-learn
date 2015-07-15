
# coding: utf-8

# In[2]:

from numpy.linalg import det
def BC(X,Y):
    return det(X.T.dot(Y)) / np.sqrt(det(X.T.dot(X)) * det(Y.T.dot(Y)))


# In[3]:

def BC_w(X,Y,M=None):
    if M==None:
        M = np.diag(np.ones(len(X)))
        
    return det(X.T.dot(M).dot(Y)) / np.sqrt(det(X.T.dot(M).dot(X)) * det(Y.T.dot(M).dot(Y)))


# In[4]:

from sklearn.decomposition import PCA


# In[5]:

import numpy as np
import numpy.random as npr


# In[6]:

npr.seed(100)
n=1000
d=3
X = npr.randn(n,d)*10


# In[7]:

Y = PCA().fit_transform(X)


# In[8]:

np.eye(10)


# In[9]:

BC(X,Y),BC_w(X,Y)


# In[ ]:

# well, that was easy


# In[1]:

X


# In[ ]:

def objective(w):
    return BC_w(X,Y,np.diag(w))

