
# coding: utf-8

# In[2]:

# goal: re-write weighted RMSD in a way that makes it easy to take derivatives w.r.t. the weight vector, or use
# gradient-free approaches


# In[3]:

import numpy.random as npr
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from autograd import grad
import autograd.numpy as np


# In[5]:

# random test data
n_atoms=10
X = np.random.randn(n_atoms,3)
Y = np.random.randn(n_atoms,3)


# In[ ]:




# In[6]:

# Kabsch algorithm-- reference: https://github.com/charnley/rmsd/blob/master/calculate_rmsd

def rotate(X,Y):
    C = np.dot(X.T, Y)
    print(C.shape)
    _,U = np.linalg.eigh(np.dot(C,C.T))
    print(U.shape)
    _,V = np.linalg.eigh(np.dot(C.T,C))
    print(V.shape)
    if (np.linalg.det(U) * np.linalg.det(V)) < 0.0:
        U[:, -1] = -U[:, -1]
    return np.dot(U,V)


# In[8]:

def rotate_w(X,Y,w=None):
    if w==None:
        w = np.ones(len(X))
    
    C = np.dot(np.dot(X.T, Y))
    _,U = np.linalg.eigh(np.dot(C,C.T))
    print(U.shape)
    _,V = np.linalg.eigh(np.dot(C.T,C))
    print(V.shape)
    if (np.linalg.det(U) * np.linalg.det(V)) < 0.0:
        U[:, -1] = -U[:, -1]
    return np.dot(U,V)


# In[9]:

import MDAnalysis.analysis.rms as rms


# In[10]:

X*np.ones((n_atoms,1))


# In[11]:

rotate(X,Y)


# In[12]:

np.random.seed(0)
triplets = [(np.random.randn(n_atoms,3),np.random.randn(n_atoms,3),np.random.randn(n_atoms,3)) for i in range(10)]


# In[13]:

from msmbuilder.example_datasets import AlanineDipeptide,FsPeptide
ala = AlanineDipeptide().get()
ala_traj = ala.trajectories[0]


# In[18]:

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(np.random.randn(100,10))
A = pca.components_.T
A.shape


# In[21]:

np.dot(A,A.T),np.dot(A.T,A)


# In[24]:

np.sum(np.abs(np.dot(A.T,A) - np.eye(2)))


# In[25]:

A.shape


# In[26]:

def orthonormal_columns_penalty(A):
    return np.sum(np.abs(np.dot(A.T,A) - np.eye(A.shape[1]))**2)


# In[29]:

orthonormal_columns_penalty(pca.components_.T)


# In[33]:

from msmbuilder.decomposition import tICA
tica = tICA(n_components=2)
tica.fit([np.random.randn(100,10)])
orthonormal_columns_penalty(tica.components_.T)


# In[47]:

comp = tica.components_.T# / tica.components_.mean(1)
comp.shape


# In[50]:

np.dot(comp.T,comp)-np.eye(2)


# In[ ]:




# In[45]:

orthonormal_columns_penalty(comp)


# In[170]:

def tripletify_trajectory(X,tau_1=5,tau_2=20,xyz=True):
    X_triplets = []
    for i in range(len(X) - tau_2):
        if xyz:
            X_triplets.append((X[i].xyz[0],X[i+tau_1].xyz[0],X[i+tau_2].xyz[0]))
        else:
            X_triplets.append((X[i],X[i+tau_1],X[i+tau_2]))

    return X_triplets


# In[177]:

ala_triplets = np.array(tripletify_trajectory(ala_traj,tau_1=5,tau_2=10),dtype=np.float64)


# In[231]:

def cheap_triplet_batch(triplets=ala_triplets,batch_size=10):
    return triplets[np.random.randint(0,len(triplets),batch_size)]


# In[1]:

def likelihood(weights,generate_triplets=cheap_triplet_batch,d=rms.rmsd):
    triplets = generate_triplets()
    return -np.sum(np.nan_to_num([d(t[0],t[1],weights) - d(t[0],t[2],weights) for t in triplets]))

def prior(weights):
    if sum(weights < 0):
        # don't accept weight vectors with any negative elements
        return -np.inf
    return 0
    #return -np.sum(weights**2)

def prob(weights,penalty_param=0.05,d=rms.rmsd):
    return likelihood(weights,d=d)+penalty_param*prior(weights)


# In[280]:

likelihood(np.ones(n_atoms))
prior(np.ones(n_atoms))


# In[281]:

likelihood(np.ones(n_atoms)),prior(np.ones(n_atoms))


# In[282]:

likelihood(np.ones(n_atoms)),likelihood(np.random.randn(n_atoms))


# In[283]:

-np.inf


# In[284]:

def RMSD(X,Y):
    # compute optimal rotation of X onto Y using SVD
    
    # compute atomwise deviations
    return


# In[285]:

get_ipython().magic(u'timeit likelihood(np.random.randn(n_atoms))')


# In[286]:

n_atoms=len(ala_triplets[0][0])
n_atoms


# In[308]:

len(p0*2)


# In[ ]:

import emcee

ndim, nwalkers = n_atoms, n_atoms*2
#p0 = [np.random.rand(ndim) for i in range(nwalkers)]

#sampler = emcee.EnsembleSampler(nwalkers, ndim, prob)
ntemps=10
sampler = emcee.PTSampler(ntemps,nwalkers,ndim,likelihood,prior)
p0 = np.random.rand(ntemps, nwalkers, ndim)
sample_results = sampler.run_mcmc(p0, 1000)


# In[320]:

sampler.flatchain.mean(0).shape


# In[321]:

sampler.flatchain.shape


# In[328]:

samples = sampler.flatchain[0,:]
samples = (samples.T / samples.sum(1)).T
samples.shape


# In[330]:

triangle.corner(samples)


# In[319]:

sampler.acceptance_fraction


# In[290]:

sample_results[0].shape


# In[291]:

sampler.flatchain.shape


# In[292]:

samples = sampler.flatchain / sampler.flatchain.mean(0)


# In[293]:

sampler.flatchain.mean(0).shape


# In[294]:

import triangle


# In[295]:

plt.plot(samples);


# In[296]:

triangle.corner(samples[:,:10])


# In[297]:

sampler.acceptance_fraction


# In[299]:

samples.mean(0),sampler.flatchain.mean(0)


# In[277]:

plt.hist(sampler.flatlnprobability,bins=50);


# In[ ]:



