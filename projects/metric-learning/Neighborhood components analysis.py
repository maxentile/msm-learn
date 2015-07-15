
# coding: utf-8

# In[1]:

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[34]:

from sklearn.datasets import load_digits
data = load_digits()
X_tot,Y_tot = load_digits().data,load_digits().target


# In[8]:

len(X_tot)


# In[9]:

split = 500
X = X_tot[:split]
Y = Y_tot[:split]
X_test = X_tot[split:]
Y_test = Y_tot[split:]


# In[10]:

from scipy.spatial.distance import pdist,squareform


# In[11]:

# stochastic neighbor assignments
def stoch_neighbor_assignments(X):
    P = squareform(np.exp(-(pdist(X)**2)))
    P -= np.diag(P)
    return np.nan_to_num(P/P.sum(1)) # columns sum to 1


# In[62]:

P = squareform(np.exp(-(pdist(X)**2)))
P -= np.diag(P)
sum(P.sum(1)==0)


# In[ ]:




# In[12]:

P = stoch_neighbor_assignments(X)
np.sum(P != 0)


# In[13]:

np.max(P),np.min(P)


# In[14]:

np.sum(np.nan_to_num(P)!=0)


# In[15]:

plt.imshow(P,interpolation='none',cmap='Blues')
plt.colorbar()


# In[16]:

# probability that point 0 will be classified correctly
sum(P.T[10,Y==Y[10]])


# In[17]:

def correct_classification_prob(P,Y):
    p = np.array([sum(P.T[i,Y==Y[i]]) for i in range(len(P))])
    return p


# In[18]:

def correct_classification_prob_vec(P,Y):
    Y_ = np.vstack([Y==y for y in Y])
    return P[Y_]


# In[19]:

def exp_class_accuracy_vectorized(P,Y):
    Y_ = np.vstack([Y==y for y in Y])
    return P[Y_].sum()/len(Y)


# In[ ]:




# In[20]:

get_ipython().magic(u'timeit sum(correct_classification_prob(P,Y))')


# In[21]:

sum(correct_classification_prob(P,Y))


# In[22]:

get_ipython().magic(u'timeit correct_classification_prob_vec(P,Y)')


# In[23]:

get_ipython().magic(u'timeit exp_class_accuracy_vectorized(P,Y)')


# In[24]:

sum(correct_classification_prob_vec(P,Y))


# In[ ]:




# In[25]:

sum(correct_classification_prob_vec(P,Y))/len(Y)


# In[26]:

get_ipython().magic(u'timeit np.vstack([Y==y for y in Y])')


# In[27]:

# expected number of points correctly classified
p = correct_classification_prob(P,Y)
correct_class_expectation = sum(p) / len(p)
correct_class_expectation


# In[28]:

plt.hist(p,bins=50);


# In[29]:

sum(p<0.5)


# In[35]:

outlier_images = data.images[p<0.5]


# In[36]:

for image in outlier_images[:5]:
    plt.figure()
    plt.imshow(-image,cmap='gray',interpolation='none')


# In[37]:

np.argmin(p)


# In[38]:

plt.imshow(-data.images[np.argmin(p)],cmap='gray',interpolation='none')


# In[39]:

sum(p==1)


# In[40]:

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_ = pca.fit_transform(X)
X_.shape


# In[41]:

P_ = stoch_neighbor_assignments(X_)
p_ = correct_classification_prob(P_,Y)
correct_class_expectation_ = sum(p_) / len(p_)
correct_class_expectation_


# In[42]:

plt.hist(p_,bins=50);


# In[43]:

# objective: find a transformation f(X) that 
# maximizes correct_classification_prob(f(X),Y)


# In[44]:

A = np.random.randn(2,X.shape[1])


# In[45]:

A.dot(X.T).shape


# In[46]:

np.sum(X.dot(A.T) - A.dot(X.T).T)


# In[47]:

X.dot(A.T),A.dot(X.T).T


# In[48]:

X_ = X.dot(A.T)
X_.shape


# In[49]:

(A.T).shape


# In[50]:

X.dot(A.T)


# In[51]:

def objective(A,X,Y):
    assert(X.shape[1]==A.shape[0])
    X_ = X.dot(A)
    P_ = stoch_neighbor_assignments(X_)
    #p_ = correct_classification_prob(P_,Y)
    #correct_class_expectation_ = sum(p_) / len(p_)
    #return correct_class_expectation_
    return exp_class_accuracy_vectorized(P_,Y)


# In[52]:

A = npr.randn(64,2)


# In[53]:

objective(A,X,Y)


# In[54]:

plt.scatter(X.dot(A)[:,0],X.dot(A)[:,1],c=Y)


# In[55]:

get_ipython().magic(u'timeit objective(A,X,Y)')


# In[56]:

get_ipython().magic(u'prun objective(A,X,Y)')


# In[276]:

# for large inputs, use ball-trees instead of computing the full P_ij matrix


# In[57]:

# construct a function we can pass to scipy optimize
def objective_vec(A):
    A_ = np.reshape(A,(X.shape[1],2))
    return objective(A_,X,Y)


# In[58]:

A = npr.randn(X.shape[1]*2)
A_ = np.reshape(A,(X.shape[1],2))
A_.shape


# In[59]:

objective_vec(npr.randn(64*2))


# In[ ]:




# In[303]:

from scipy.optimize import minimize,basinhopping


# In[296]:

A_init = pca.components_.T
A_init.shape


# In[298]:

objective(A_init,X,Y)


# In[299]:

A_init_vec = np.reshape(A_init,np.prod(A_init.shape))


# In[301]:

obj_min = lambda A:-objective_vec(A)


# In[306]:

obj_min(A_init_vec)


# In[307]:

res = minimize(obj_min,A_init_vec,options={'maxiter':2,'disp':True})


# In[310]:

def gradient(func,x0,h=0.001):
    x0 = np.array(x0)#,dtype=float)
    y = func(x0)
    deriv = np.zeros(len(x0))
    for i in range(len(x0)):
        x = np.array(x0)
        x[i] += h
        deriv[i] = (func(x) - y)/h
    return deriv


# In[311]:

get_ipython().magic(u'timeit obj_min(A_init_vec)')


# In[312]:

len(A_init_vec)


# In[313]:

gradient(obj_min,A_init_vec)


# In[ ]:

def obj_grad(A,X,Y)


# In[ ]:

def gradient(A,X,Y):
    X_ = X.dot(A)
    P_ = stoch_neighbor_assignments(X_)
    s = 0
    for i in range(len(X_)):
        s+=
    return 


# In[302]:

res = basinhopping(obj_min,A_init_vec,disp=True)


# In[308]:

from scipy.optimize import minimize, rosen, rosen_der


# In[309]:

x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method='Nelder-Mead')
res.x


# In[63]:

from autograd import grad
import autograd.numpy as np


# In[64]:

grad(objective_vec)(npr.randn(64*2))


# In[ ]:



