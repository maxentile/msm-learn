
# coding: utf-8

# In[2]:

from autograd import grad
import autograd.numpy as np


# In[2]:

from numpy.linalg import det
def BC(X,Y):
    return det(np.dot(X.T,Y)) / np.sqrt(det(np.dot(X.T,X)) * det(np.dot(Y.T,Y)))


# In[23]:

np.random.seed(0)
X = np.random.rand(1000)*10
Y = 2*X + np.random.randn(1000)+5


# In[ ]:




# In[3]:

import matplotlib.pyplot as plt


# In[4]:

get_ipython().magic(u'matplotlib inline')


# In[26]:

plt.scatter(X,Y)


# In[104]:

def loss(theta):
    return -np.sum((Y-(theta[0]*X+theta[1]))**2)


# In[105]:

loss((2,5))


# In[106]:

gradient = grad(loss)


# In[107]:

gradient(np.zeros(2))


# In[111]:

n=1000
x = np.zeros((n,2))
for i in range(1,n):
    x[i] = x[i-1] + gradient(x[i-1])*0.000001


# In[112]:

x[-1]


# In[113]:

plt.plot(x[:,0],x[:,1])
plt.scatter(x[:,0],x[:,1])


# In[446]:

def autocorrelation(X,k=1):
    mu = X.mean(0)
    denom=(len(X)-k)*np.std(X,0)**2
    s = np.sum((X[:-k]-mu)*(X[k:]-mu),0)
    return np.sum(s/denom)
    #return np.sum(s/denom)


# In[531]:

def time_lag_corr_cov(X,tau=1):
    #mu = (X[:-tau].mean(0) + X[tau:].mean(0)) / 2
    mu = X.mean(0)
    X_ = X-mu
    M = len(X) - tau
    dim = len(X.T)
    corr = np.zeros((dim,dim))
    cov = np.zeros((dim,dim))
    for i in range(M):
        corr += np.outer(X_[i],X_[i+tau]) + np.outer(X_[i+tau],X_[i])
        cov += np.outer(X_[i],X_[i]) + np.outer(X_[i+tau],X_[i+tau])
    return corr / (2.0*M),cov / (2.0*M)


# In[536]:

def autocorr(X,tau=1):
    mu = X.mean(0)
    X_ = X-mu
    M = len(X) - tau
    dim = len(X.T)
    corr = np.zeros((dim,dim))
    for i in range(M):
        corr += np.outer(X_[i],X_[i+tau]) + np.outer(X_[i+tau],X_[i])
    return corr / (2.0*M)

c = autocorr(X_dihedral[:10000])
plt.imshow(c,interpolation='none')


# In[549]:

plt.hist(c.reshape(np.prod(c.shape)),bins=50);


# In[553]:

for i in range(10):
    print(np.sum(np.abs(autocorr(np.random.randn(1000,84)))))


# In[552]:

np.sum(np.abs(autocorr(X_dihedral[:10000])))


# In[532]:

time_lag_corr_cov(X_dihedral)


# In[447]:

np.std(X_dihedral,0).shape


# In[448]:

X_dihedral.mean(0).shape


# In[5]:

from msmbuilder.example_datasets import AlanineDipeptide,FsPeptide
dataset = FsPeptide().get()
fs_trajectories = dataset.trajectories
from msmbuilder import featurizer
dhf = featurizer.DihedralFeaturizer()
dhft = dhf.fit_transform(fs_trajectories)
X_dihedral = np.vstack(dhft)#[0]


# In[508]:

X_dihedral.mean(0).shape


# In[509]:

X_dihedral.shape


# In[510]:

autocorrelation(X_dihedral)


# In[511]:

from sklearn.decomposition import PCA
pca = PCA(2)
autocorrelation(pca.fit_transform(X_dihedral))


# In[513]:

X_ = pca.fit_transform(X_dihedral)
plt.scatter(X_[:,0],X_[:,1],linewidths=0,s=1,
            c=np.arange(len(X_)),alpha=0.5)


# In[514]:

A_init = pca.components_.T
A_init.shape


# In[515]:

np.dot(X_dihedral,A_init)


# In[8]:

from msmbuilder.decomposition import tICA
tica = tICA(2,10)
X_tica = tica.fit_transform([X_dihedral])[0]
#autocorrelation(X_tica)


# In[519]:

plt.scatter(X_tica[:,0],X_tica[:,1],linewidths=0,s=4,
            c=np.arange(len(X_)),alpha=0.5)


# In[520]:

A_init_tica=tica.components_.T


# In[521]:

def autocorr_loss(A_vec):
    A = np.reshape(A_vec,A_init.shape)
    X_ = np.dot(X_dihedral,A)
    X_ /= (np.max(X_) - np.min(X_))
    return autocorrelation(X_)


# In[522]:

autocorr_loss(A_init_tica.reshape(84*2))


# In[523]:

autocorr_grad = grad(autocorr_loss)


# In[524]:

plt.hist(autocorr_grad(A_init_tica.reshape(84*2)));


# In[525]:

plt.hist(A_init_tica.reshape(84*2),bins=50);


# In[411]:

get_ipython().magic(u'timeit autocorr_loss(A_init_tica.reshape(84*2))')


# In[412]:

get_ipython().magic(u'timeit autocorr_grad(A_init_tica.reshape(84*2))')


# In[528]:

n=100
x = np.zeros((n,84*2))
x[0] = A_init_tica.reshape(84*2)
from time import time
t = time()
for i in range(1,n):
    x[i] = x[i-1] + autocorr_grad(x[i-1])*10
    print(i,time()-t)


# In[560]:

plt.plot(x);


# In[530]:

X_ = np.dot(X_dihedral,x[-1].reshape(84,2))
plt.scatter(X_[:,0],X_[:,1],linewidths=0,s=4,
            c=np.arange(len(X_)),alpha=0.5)


# In[561]:

X_dihedral.shape


# In[ ]:




# In[468]:

for i in range(len(x))[::50]:
    X_ = np.dot(X_dihedral,x[i].reshape(84,2))
    plt.scatter(X_[:,0],X_[:,1],linewidths=0,s=4,
                c=np.arange(len(X_)),alpha=0.5)
    plt.savefig('{0}.jpg'.format(i))
    plt.close()


# In[469]:

autocorr_loss(x[-1]),autocorr_loss(A_init_tica),autocorr_loss(A_init)


# In[478]:

def autocorr_loss_mult(A_vec):
    A = np.reshape(A_vec,A_init.shape)
    X_ = np.dot(X_dihedral,A)
    return autocorrelation(X_,1) + autocorrelation(X_,10)
    #s = 0
    #for i in range(10):
    #    s += autocorrelation(X_,1+2*i)
    return autocorrelation(X_)


# In[479]:

autocorrelation(X_dihedral,10)


# In[480]:

autocorr_grad_mult = grad(autocorr_loss_mult)


# In[482]:

autocorr_loss_mult(np.ones(84*2))


# In[481]:

autocorr_grad_mult(np.ones(84*2)).shape


# In[486]:

n=1000
x = np.zeros((n,84*2))
x[0] = A_init_tica.reshape(84*2)
for i in range(1,n):
    x[i] = x[i-1] + autocorr_grad_mult(x[i-1])


# In[487]:

plt.plot(x);


# In[495]:

X_ = np.dot(X_dihedral,x[100].reshape(84,2))
plt.scatter(X_[:,0],X_[:,1],linewidths=0,s=4,
            c=np.arange(len(X_)),alpha=0.5)


# In[490]:

l = [autocorr_loss_mult(x_) for x_ in x]


# In[492]:

l


# In[491]:

plt.plot(l)


# In[ ]:

# idea: tICA requires the specification of 
# a single autocorrelation time-- can we consider multiple?


# In[563]:

# we want to find an embedding of the dihedral angles that puts kinetically-nearby points near each other in the embedding


# In[564]:

X_dihedral.shape


# In[653]:

pca_w = PCA(whiten=True)
X_dihedral_whitened = pca_w.fit_transform(X_dihedral)


# In[654]:

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_dihedral_whitened)


# In[655]:

sum(pca.explained_variance_ratio_)


# In[656]:

from scipy.spatial.distance import euclidean


# In[906]:

def d(x,y):
    return np.sqrt(np.dot(x-y,x-y))

def scalar_penalize(close_distance,far_distance):
    return close_distance-far_distance

def mult_penalize(close_distance,far_distance):
    return close_distance/far_distance

def exp_penalize(close_distance,far_distance,scale=10):
    return np.exp(scale*(close_distance-far_distance))

def zero_one_penalize(close_distance,far_distance):
    return 1.0*(close_distance > far_distance)

def triplet_batch_objective_simple(embedding_points,tau_1=1,tau_2=10,penalize=scalar_penalize):
    loss = 0.0
    n_triplets = len(embedding_points) - tau_2
    assert(n_triplets>0)
    for i in range(n_triplets):
        close = d(embedding_points[i],embedding_points[i+tau_1])
        far = d(embedding_points[i],embedding_points[i+tau_2])
        loss += penalize(close,far)
        #print(close,far)
        #print(contribution)
    return loss / n_triplets


# In[836]:

triplet_batch_objective_simple(X_dihedral,penalize=zero_one_penalize)


# In[837]:

triplet_batch_objective_simple(X_dihedral,tau_1=5,tau_2=10,penalize=zero_one_penalize)


# In[839]:

triplet_batch_objective_simple(X_dihedral,tau_1=1,tau_2=100,penalize=zero_one_penalize)


# In[840]:

sample = X_dihedral[:10000]


# In[842]:

get_ipython().magic(u'timeit triplet_batch_objective_simple(sample,tau_1=1,tau_2=100,penalize=zero_one_penalize)')


# In[861]:

taus = np.array([1,2,3,4,5,10,20,30,40,50,100,200,300,400,500])
results = np.zeros((len(taus),len(taus)))

for i,tau_1 in enumerate(taus):
    for j,tau_2 in enumerate(taus):
        if tau_2 > tau_1:
            results[i,j] = triplet_batch_objective_simple(sample,tau_1=tau_1,tau_2=tau_2,penalize=zero_one_penalize)


# In[864]:

plt.imshow(results,interpolation='none',cmap='Blues')
plt.colorbar()


# In[718]:

# alternate flow: select random center points along the trajectory....


# In[907]:

def stoch_triplet_objective(transform,full_set,tau_1=1,tau_2=10,batch_size=50,penalize=scalar_penalize):
    
    '''to-do: make this work with a list of trajectories'''
    
    if type(full_set)==list:
        # it's a list of trajectories, each a numpy array
        list_ind = np.random.randint(0,len(full_set),batch_size)
        centers = np.random.randint(0,len(full_set[0])-tau_2,batch_size)
        triplets = [(full_set[l][c],full_set[l][c+tau_1],full_set[l][c+tau_2]) for l,c in zip(list_ind,centers)]
    else:
        # it's just one trajectory in a numpy array
        centers = np.random.randint(0,len(full_set)-tau_2,batch_size)
        triplets = [(full_set[c],full_set[c+tau_1],full_set[c+tau_2]) for c in centers]
    
    triplets = [(transform(a),transform(b),transform(c)) for (a,b,c) in triplets]
    loss = 0
    for i in range(batch_size):
        close = d(triplets[i][0],triplets[i][1])
        far = d(triplets[i][0],triplets[i][2])
        loss += penalize(close,far)
    return loss / batch_size


# In[775]:

get_ipython().magic(u'timeit stoch_triplet_objective(lambda i:i,dhft,batch_size=100)')


# In[787]:

loss(pca.components_.T)


# In[900]:

A.shape,A.sum(0).shape
sum(np.abs(A)),sum(np.abs(A/np.abs(A).sum(0)))


# In[910]:

def loss(weights,batch_size=1000):
    transform = lambda x:np.dot(x,weights)
    return stoch_triplet_objective(transform,dhft,batch_size=batch_size)

def loss_vec(weights,target_dim=2,batch_size=100,penalize=mult_penalize):
    A = np.reshape(weights,(weights.shape[0]/target_dim,target_dim))
    #A /= np.sum(A,0)
    #A /= np.abs(A).sum(0)
    #A = A/np.reshape(np.sum(A**2,1),(weights.shape[0]/target_dim,1))
    transform = lambda x:np.dot(x,A)
    return stoch_triplet_objective(transform,dhft,batch_size=batch_size,penalize=penalize)

grad_loss = grad(loss)
plt.scatter(grad_loss(np.ones(84)),grad_loss(np.ones(84)))
plt.hlines(0,-0.1,0.1,linestyles='--')
plt.vlines(0,-0.1,0.1,linestyles='--')

print(spearmanr(grad_loss(np.ones(84)),grad_loss(np.ones(84))))

plt.figure()

grad_loss = grad(loss_vec)
plt.scatter(grad_loss(np.ones(84*2)),grad_loss(np.ones(84*2)))
plt.hlines(0,-0.1,0.1,linestyles='--')
plt.vlines(0,-0.1,0.1,linestyles='--')
spearmanr(grad_loss(np.ones(84*2)),grad_loss(np.ones(84*2)))


# In[829]:

from scipy.optimize import minimize
from autograd.convenience_wrappers import hessian_vector_product as hvp
results = minimize(lambda w:loss_vec(w,batch_size=1000),pca.components_.T.flatten(),jac=grad(loss_vec),hessp=hvp(loss_vec,84*2))


# In[825]:

results


# In[826]:

A = results['x'].reshape((84,2))
A = A/np.reshape(np.sum(A**2,1),(84,1))
projected = np.dot(X_dihedral,A)
projected /= np.sum(projected,0)
plt.scatter(projected[:,0],projected[:,1],linewidths=0,s=4,
            c=np.arange(len(projected)),alpha=0.5)


# In[809]:

np.sum(A,0).shape


# In[802]:

triplet_batch_objective_simple(projected),triplet_batch_objective_simple(X_pca),triplet_batch_objective_simple(X_tica)


# In[877]:

def adagrad(grad, x, num_iters=100, step_size=0.1, gamma=0.9, eps = 10**-8):
    """Root mean squared prop: See Adagrad paper for details.
    
    Stolen from autograd examples: https://github.com/HIPS/autograd/blob/master/examples/optimizers.py#L21"""
    avg_sq_grad = np.ones(len(x))
    history = np.zeros((num_iters+1,len(x)))
    history[0] = x
    for i in xrange(num_iters):
        g = grad(x)
        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x -= step_size * g/(np.sqrt(avg_sq_grad) + eps)
        history[i+1] = x
    return history


# In[920]:

loss_func = lambda weights:loss_vec(weights,batch_size=1000,penalize=scalar_penalize)
history = adagrad(grad(loss_func),pca.components_.T.flatten(),num_iters=100)


# In[929]:

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_dihedral)


# In[921]:

normed_history = norm_history(history)
plt.plot(normed_history[:,:10]);


# In[932]:

def rotation_matrix(theta=np.pi/2):
    r = np.zeros((2,2))
    r[0,0] = r[1,1] = np.cos(theta)
    r[0,1] = np.sin(theta)
    r[1,0] = -np.sin(theta)
    return r

np.dot(projected,rotation_matrix())


# In[935]:

proj_mat = np.reshape(normed_history[-1],(84,2))
projected = np.dot(X_dihedral,proj_mat)
#projected /= np.sum(projected,0)
projected = np.dot(projected,rotation_matrix(np.pi/4))
plt.scatter(projected[:,0],projected[:,1],linewidths=0,s=1,
            c=np.arange(len(projected)),alpha=0.5,cmap='rainbow')
plt.title('Triplet-based linear embedding')
plt.figure()

plt.scatter(X_pca[:,0],X_pca[:,1],linewidths=0,s=1,
            c=np.arange(len(projected)),alpha=0.5,cmap='rainbow')
plt.title('PCA')
plt.figure()

plt.scatter(X_tica[:,0],X_tica[:,1],linewidths=0,s=1,
            c=np.arange(len(projected)),alpha=0.5,cmap='rainbow')
plt.title('tICA')


# In[973]:

from mdp.nodes import XSFANode
xsfa = XSFANode(output_dim=2)
pca = PCA()
X_dihedral_ = pca.fit_transform(X_dihedral)[:,:20]
X_xsfa = xsfa.execute(X_dihedral_)


# In[975]:

len(projected),len(X_xsfa)


# In[963]:

plt.plot(pca.explained_variance_ratio_)


# In[976]:

plt.scatter(X_xsfa[:,0],X_xsfa[:,1],linewidths=0,s=1,
            c=np.arange(len(X_xsfa)),alpha=0.5,cmap='rainbow')


# In[ ]:

# now let's test which embedding is best for constructing MSMs...

from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from sklearn.pipeline import Pipeline

results = dict()
embeddings = [('tICA',X_tica),('PCA',X_pca),('Triplet',projected),('xSFA',X_xsfa)]
for (name,dataset) in embeddings:
    pipeline = Pipeline([
        ('cluster', MiniBatchKMeans(n_clusters=100)),
        ('msm', MarkovStateModel(lag_time=10))
    ])
    pipeline.fit([dataset[:100000]])
    results[name] = pipeline
    print(name,pipeline.score([dataset[100000:]]))


# In[951]:

msm = MarkovStateModel()
msm.fit(np.random.randint(0,10,100000))
print(msm.summarize())


# In[916]:

from time import time
t = time()
grad(loss_vec)(pca.components_.T.flatten())
print(time()-t)


# In[873]:

pca.components_.T.flatten().shape


# In[786]:

from scipy.stats import spearmanr
spearmanr(grad_loss(np.ones(84)),grad_loss(np.ones(84)))


# In[ ]:




# In[749]:

objective_evaluations = np.array([stoch_triplet_objective(dhft,tau_1=1,tau_2=100,batch_size=100) for _ in range(1000)])


# In[750]:

objective_evaluations.std(),objective_evaluations.mean()


# In[743]:

fs_trajectories


# In[742]:

a = []
type(a)==list


# In[738]:

objective_evaluations = np.array([stoch_triplet_objective(X_pca,tau_1=1,tau_2=100,batch_size=100) for _ in range(1000)])


# In[739]:

objective_evaluations.std(),objective_evaluations.mean()


# In[658]:

pca.components_.shape


# In[659]:

triplet_batch_objective_simple(X_pca[:11])


# In[660]:

np.dot(X_dihedral,pca.components_.T).shape


# In[661]:

def sgd(objective,dataset,init_point,batch_size=20,n_iter=100,step_size=0.01,seed=0,stoch_select=False):
    ''' objective takes in a parameter vector and an array of data'''
    np.random.seed(seed)
    testpoints = np.zeros((n_iter,len(init_point)))
    testpoints[0] = init_point
    ind=0
    for i in range(1,n_iter):
        if stoch_select:
            
        else:
            max_ind = ind+batch_size
            if max_ind>=len(dataset):
                ind = max_ind % len(dataset)
                max_ind = ind+batch_size
            subset = dataset[ind:max_ind]
            ind = (ind + batch_size)
        obj_grad = grad(lambda p:objective(p,subset))
        raw_grad = obj_grad(testpoints[i-1])
        gradient = np.nan_to_num(raw_grad)
        #print(gradient,raw_grad)
        testpoints[i] = testpoints[i-1] - gradient*step_size
    return np.array(testpoints)


# In[662]:

def projection_obj(proj_vec,subset):
    # WARNING: CURRENTLY HARD-CODED PROJECTION MATRIX DIMENSIONS...
    A = np.reshape(proj_vec,(84,2))
    A /= (A**2).sum(0)
    projected = np.dot(subset,A)
    return triplet_batch_objective_simple(projected)


# In[663]:

(A**2).sum(0)


# In[ ]:




# In[712]:

raw_points = sgd(projection_obj,X_dihedral_whitened,pca.components_.T.flatten(),step_size=0.01,
                 n_iter=1000,batch_size=20)


# In[881]:

def norm(s):
    return s / np.sqrt(np.sum(s**2))

plt.plot(raw_points[:,:10]);
plt.figure()
def norm_history(raw_points):
    return np.array([norm(s) for s in raw_points])
normed_points = norm_history(raw_points)
plt.plot(normed_points);


# In[707]:

(pca.components_).flatten().shape


# In[708]:

A = np.reshape(raw_points[-1],(84,2))
projected = np.dot(X_dihedral,A)


# In[709]:

plt.scatter(projected[:,0],projected[:,1],linewidths=0,s=4,
            c=np.arange(len(projected)),alpha=0.5)


# In[681]:

plt.plot(np.linalg.eigh(A.dot(A.T))[0][-5:])


# In[672]:

tica = tICA(n_components=2)
X_tica = tica.fit_transform([X_dihedral_whitened])[0]


# In[673]:

plt.scatter(X_tica[:,0],X_tica[:,1],linewidths=0,s=4,
            c=np.arange(len(projected)),alpha=0.5)


# In[674]:

triplet_batch_objective_simple(X_pca),triplet_batch_objective_simple(X_tica),triplet_batch_objective_simple(projected)


# In[9]:

tica = tICA(n_components=2)


# In[76]:

from sklearn.preprocessing import PolynomialFeatures
from time import time
t = time()
dhft_poly = []
poly = PolynomialFeatures()
for i in range(len(dhft)):
    dhft_poly.append(poly.fit_transform(dhft[i]))
    print(i,time()-t)


# In[ ]:




# In[15]:

dhft[0].shape


# In[19]:

get_ipython().magic(u'timeit poly.fit_transform(dhft[0][:1000])')


# In[24]:

dhft_poly_0 = poly.fit_transform(dhft[0])


# In[25]:

dhft_poly_0.shape


# In[48]:

tica=tICA(n_components=2,lag_time=10)


# In[80]:

X_tica_poly = tica.fit_transform(dhft_poly)


# In[81]:

X_tica_poly_vstack = np.vstack(X_tica_poly)


# In[84]:

plt.scatter(X_tica_poly_vstack[:,0],X_tica_poly_vstack[:,1],linewidths=0,s=1,
            #c=np.arange(len(X_tica_poly_vstack)),
            c=np.vstack([i*np.ones(len(X_tica_poly[0])) for i in range(len(X_tica_poly))]),
            alpha=0.5,cmap='rainbow')
plt.title('Nonlinear tICA')


# In[78]:

tica = tICA(n_components=2,lag_time=10)
X_tica = tica.fit_transform(dhft)


# In[79]:

X_tica_vstack = np.vstack(X_tica)
plt.scatter(X_tica_vstack[:,0],X_tica_vstack[:,1],linewidths=0,s=1,
            c=np.arange(len(X_tica_vstack)),alpha=0.5,cmap='rainbow')
plt.title('tICA')


# In[87]:

from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from sklearn.pipeline import Pipeline



results = dict()
embeddings = [('tICA',X_tica),('Nonlinear tICA',X_tica_poly)]
for (name,dataset) in embeddings:
    pipeline = Pipeline([
        ('cluster', MiniBatchKMeans(n_clusters=100)),
        ('msm', MarkovStateModel(lag_time=1))
    ])
    #pipeline.fit([dataset[:5000]])
    #pipeline.fit(dataset)
    
    pipeline.fit(dataset[:14])
    
    results[name] = pipeline
    #print(pipeline.steps[1][1].score_)
    
    print(pipeline.score(dataset[14:]))
    
    #print(name,pipeline.score([dataset[5000:]]))


# In[67]:

msm = MarkovStateModel()
msm.score_


# In[56]:

X_tica_poly.shape


# In[ ]:



