
# coding: utf-8

# In[1]:

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from autograd import grad
import autograd.numpy as np

from sklearn.datasets import load_digits
data = load_digits()
X,Y = load_digits().data,load_digits().target
X.shape


# In[2]:

# in a triplet network, the loss is defined on triplets of observations


# In[3]:

def triplet_loss_paper(distance_close,distance_far):
    ''' loss function given two distances from the triplet (a,b,c)
    where distance_close=d(a,b), the distance between two points that
    should be close, and distance_far=d(a,c), the 
    distance between two points that should be far.
    
    see page 3 of: http://arxiv.org/abs/1412.6622'''
    exp_close = np.exp(distance_close)
    exp_far = np.exp(distance_close)
    
    
    d_plus = exp_close / (exp_close + exp_far)
    d_minus = exp_far / (exp_close + exp_far)
    
    return (d_plus-d_minus)**2

def modified_triplet_loss(distance_close,distance_far):
    return distance_far - distance_close

def zero_one_triplet_loss(distance_close,distance_far):
    return 1.0*(distance_far < distance_close)

def distance(x,y):
    return np.sqrt(np.sum((x-y)**2))
                 
def triplet_objective(transformation,x,x_close,x_far,
                      triplet_loss=modified_triplet_loss):
    '''for a metric embedding, we can transform each point separately'''
    t_a,t_b,t_c = [transformation(point) for point in (x,x_close,x_far)]
    
    return triplet_loss(distance(t_a,t_b),distance(t_a,t_c))

def generate_triplet(X,Y):
    x_ind = np.random.randint(len(X))
    x,y = X[x_ind],Y[x_ind]
    x_close_ind = np.random.randint(len(X[Y==y]))
    x_far_ind = np.random.randint(len(X[Y!=y]))
    x_close = X[Y==y][x_close_ind]
    x_far = X[Y!=y][x_far_ind]
    return x,x_close,x_far


# In[14]:

get_ipython().magic(u'timeit generate_triplet(X,Y)')


# In[4]:

from time import time
t = time()
triplets = [generate_triplet(X[:1000],Y[:1000]) for i in range(10000)]
triplets_test = [generate_triplet(X[1000:],Y[1000:]) for i in range(10000)]
print(time() - t)


# In[5]:

triplets = np.array(triplets)
triplets.shape


# In[6]:

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)
transform = lambda x: pca.transform(x)[0]


# In[7]:

triplet_objective(pca.transform,*triplets[10])


# In[8]:

t =time()
obj = [triplet_objective(transform,*triplet) for triplet in triplets]
print(time() - t)


# In[9]:

np.min(obj),np.max(obj)


# In[142]:

def mahalanobis(x,y,diag):
    return np.sqrt(np.dot((x-y)*diag,(x-y)))
    #return np.sqrt(np.sum(np.abs(np.dot(np.outer((x-y),diag),(x-y)))))
    #return np.sqrt(np.dot(np.outer((x-y),np.diag(diag)),x-y))


# In[10]:

def weighted_metric(x,y,a,W):
    ''' a is a non-negative vector of length len(x)=len(y), w is a len(x)-by-len(x) real matrix'''
    return np.sqrt(np.dot(np.dot(np.dot(np.dot((x-y).T,np.diag(a)),W),np.diag(a).T),(x-y)))


# In[13]:

weighted_metric(np.ones(2),np.ones(2)*2,np.ones(2),np.ones((2,2)))


# In[14]:

get_ipython().magic(u'timeit weighted_metric(np.ones(2),np.ones(2)*2,np.ones(2),np.ones((2,2)))')


# In[ ]:




# In[143]:

mahalanobis(np.ones(2),np.ones(2),np.ones(2))


# In[ ]:

pca.transform(


# In[146]:

mahalanobis(pca.transform(X[0])[0],pca.transform(X[1])[0],np.ones(2))


# In[47]:

pca.transform(X[0])


# In[147]:

def mahalanobis_obj(weights,triplet,triplet_loss=modified_triplet_loss):
    return triplet_loss(mahalanobis(triplet[0],triplet[1],weights),
                        mahalanobis(triplet[0],triplet[2],weights))

def batch_mahalanobis_obj(weights,triplets,triplet_loss=modified_triplet_loss):
    loss = 0
    for i in range(len(triplets)):
        loss += mahalanobis_obj(weights,triplets[i],triplet_loss)
    return loss / len(triplets)


print(mahalanobis_obj(np.ones(len(triplets[0][0])),triplets[0]))
print(batch_mahalanobis_obj(np.ones(len(triplets[0][0])),triplets[:100]))


# In[148]:

grad(lambda w:mahalanobis_obj(w,triplets[0]))(np.ones(len(triplets[0][0])))


# In[547]:

def sgd(objective,dataset,init_point,n_iter=100,step_size=0.01,seed=0,
        stoch_select=False,norm=False,store_intermediates=True):
    ''' objective takes in a parameter vector and an array of data'''
    np.random.seed(seed)
    x = init_point
    if store_intermediates:
        testpoints = np.zeros((n_iter,len(init_point)))
        testpoints[0] = x

    for i in range(1,n_iter):
        ind = np.random.randint(len(dataset))
        obj_grad = grad(lambda p:objective(p,dataset[ind]))
        raw_grad = obj_grad(x)
        #raw_grad = obj_grad(testpoints[i-1])
        gradient = np.nan_to_num(raw_grad)
        x = x-gradient*step_size
        if norm:
            x = np.abs(x) / np.sum(np.abs(x))
        #print(gradient,raw_grad)
        if store_intermediates:
            testpoints[i] = x
    return np.array(testpoints)


# In[173]:

results = sgd(mahalanobis_obj,triplets,np.ones(len(triplets[0][0])),n_iter=10000,step_size=0.001)
plt.plot(results[:,:10]);


# In[174]:

plt.plot(results);


# In[175]:

len(triplets[0][0])


# In[176]:

batch_mahalanobis_obj(np.ones(64),triplets,zero_one_triplet_loss)


# In[177]:

batch_mahalanobis_obj(results[-1],triplets,zero_one_triplet_loss)


# In[178]:

batch_mahalanobis_obj(np.ones(64),triplets_test,zero_one_triplet_loss),batch_mahalanobis_obj(results[-1],triplets_test,zero_one_triplet_loss)


# In[179]:

grad(lambda w:batch_mahalanobis_obj(w,triplets[:100]))(np.ones(len(triplets[0][0])))


# In[121]:

cheap_triplet_batch = lambda batch_size=50: triplets[np.random.randint(0,len(triplets),batch_size)]


# In[66]:

get_ipython().magic(u'timeit cheap_triplet_batch()')


# In[134]:

example_triplet_batch = cheap_triplet_batch()
#stoch_mahalanobis_grad = grad(lambda weights:batch_mahalanobis_obj(weights,cheap_triplet_batch()))
mahalanobis_grad = grad(lambda weights:batch_mahalanobis_obj(weights,example_triplet_batch))


# In[180]:

mahalanobis_grad(np.ones(64))


# In[181]:

from scipy.optimize import minimize

result = minimize(lambda w:batch_mahalanobis_obj(w,cheap_triplet_batch()),np.ones(64),jac=mahalanobis_grad)


# In[83]:

result


# In[85]:

plt.plot(result.x)


# In[118]:

mahalanobis(triplets[0][0],triplets[0][1],result.x)


# In[109]:

x,y=triplets[0][1],triplets[0][0]
np.sum(np.abs(np.dot(np.outer((x-y),result.x),(x-y))))


# In[106]:

np.outer(x-y,x-y).shape


# In[117]:

batch_mahalanobis_obj(result.x,triplets[:10])


# In[2]:

def feedforward_network


# In[5]:

import keras


# In[182]:

def objective(M):
    return np.sum(np.abs(M))

grad(objective)(np.ones((10,10)))


# In[183]:

def tripletify_trajectory(X,tau_1=5,tau_2=20):
    X_triplets = []
    for i in range(len(X) - tau_2):
        X_triplets.append((X[i],X[i+tau_1],X[i+tau_2]))
    return X_triplets


# In[186]:

X_ = tripletify_trajectory(X)


# In[198]:

def deviation_ify_triplets(triplets):
    deviations = np.zeros((len(triplets),2,len(triplets[0][0])))
    for i,(a,b,c) in enumerate(triplets):
        deviations[i][0] = a-b
        deviations[i][1] = a-c
    return deviations


# In[199]:

deviations = deviation_ify_triplets(X_)


# In[324]:

def weighted_objective(weights,deviation_pair,norm=False):
    close = np.dot(deviation_pair[0],np.abs(weights))
    far = np.dot(deviation_pair[1],np.abs(weights))
    if norm:
        return (close-far) / (close+far)
    else:
        return close-far


# In[289]:

weighted_objective(np.ones(64),deviations[0])


# In[290]:

stoch_objective = lambda weights:weighted_objective(weights,deviations[np.random.randint(len(deviations))])


# In[291]:

stoch_objective(np.ones(64))


# In[334]:

results = sgd(weighted_objective,deviations,np.ones(64),n_iter=10000,step_size=0.001)


# In[335]:

plt.plot(results);


# In[336]:

from msmbuilder.example_datasets import AlanineDipeptide,FsPeptide
ala = AlanineDipeptide().get()
ala_traj = ala.trajectories[0]
fs = FsPeptide().get()
fs_traj = fs.trajectories[0]


# In[295]:

ala_traj


# In[296]:

from msmbuilder.featurizer import DihedralFeaturizer
dhf = DihedralFeaturizer()
dhft = dhf.transform([ala_traj])[0]


# In[297]:

dhft.shape


# In[364]:

deviations_ala = deviation_ify_triplets(tripletify_trajectory(dhft,tau_1=5,tau_2=10))


# In[365]:

results = sgd(weighted_objective,deviations_ala,np.ones(4),n_iter=10000,step_size=0.01)


# In[366]:

plt.plot(results);


# In[273]:

import sys
sys.path.append('../projects/metric-learning')
import weighted_rmsd


# In[301]:

from weighted_rmsd import compute_atomwise_deviation,compute_atomwise_deviation_xyz


# In[367]:

ala_triplets = tripletify_trajectory(ala_traj,tau_1=5,tau_2=10)
fs_triplets = tripletify_trajectory(fs_traj,tau_1=5,tau_2=10)


# In[303]:

a = ala_triplets[0][0]
a.n_atoms


# In[304]:

def deviation_ify_protein_triplets(triplets):
    deviations = np.zeros((len(triplets),2,triplets[0][0].n_atoms))
    for i,(a,b,c) in enumerate(triplets):
        deviations[i][0] = compute_atomwise_deviation(a,b)
        deviations[i][1] = compute_atomwise_deviation(a,c)
    return deviations


# In[514]:

ala_deviations = deviation_ify_protein_triplets(ala_triplets)
fs_deviations = deviation_ify_protein_triplets(fs_triplets)


# In[515]:

fs_atoms = fs_deviations.shape[-1]


# In[516]:

results_ala = sgd(weighted_objective,ala_deviations,np.ones(22),n_iter=10000,step_size=0.1)
results_fs = sgd(weighted_objective,fs_deviations,np.ones(fs_atoms),n_iter=10000,step_size=0.1)


# In[517]:

norm_results_ala = 22*results_ala / np.outer(results_ala.sum(1),np.ones(results_ala.shape[1]))
norm_results_fs = fs_atoms*results_fs / np.outer(results_fs.sum(1),np.ones(fs_atoms))


# In[518]:

plt.plot(norm_results_ala);
plt.figure()
plt.plot(norm_results_fs);


# In[519]:

def zero_one_weighted_deviation_loss(weights,deviation_pair):
    return 1.0*(np.dot(deviation_pair[0],weights) > np.dot(deviation_pair[1],weights))


# In[520]:

sum([zero_one_weighted_deviation_loss(np.ones(22),a) for a in ala_deviations]) / len(ala_deviations)


# In[521]:

sum([zero_one_weighted_deviation_loss(np.abs(results_ala[-1]),a) for a in ala_deviations]) / len(ala_deviations)


# In[522]:

sum([zero_one_weighted_deviation_loss(np.ones(fs_atoms),a) for a in fs_deviations]) / len(fs_deviations)


# In[523]:

sum([zero_one_weighted_deviation_loss(np.abs(results_fs[-1]),a) for a in fs_deviations]) / len(fs_deviations)


# In[363]:

from weighted_rmsd import compute_kinetic_weights


# In[ ]:

compute_kinetic_weights(tra


# In[319]:

print('residues with increased weight here: ',np.arange(len(norm_results))[norm_results[-1]>1])


# In[ ]:




# In[372]:

from MDAnalysis.analysis import align


# In[484]:

a = np.array(fs_traj[0].xyz[0],dtype=np.float64)
b = np.array(fs_traj[1].xyz[0],dtype=np.float64)
a


# In[410]:

align.rms.rmsd(a,b,np.ones(len(a)))


# In[412]:

grad(lambda w:align.rms.rmsd(a,b,w))(np.ones(len(a)))


# In[413]:

get_ipython().magic(u'timeit align.rms.rmsd(a,b)')


# In[384]:

align.rms.rmsd(a,a,10*np.ones(fs_atoms))


# In[ ]:




# In[385]:

import numpy.linalg as la


# In[389]:

u,s,v=la.svd(np.ones((10,10)))


# In[396]:

def can_has_svd_derivative(m):
    u,s,v=la.svd(m)
    return np.sum(u)


# In[400]:

plt.hist([can_has_svd_derivative(np.random.randn(10,10)) for i in range(10000)],bins=50);


# In[401]:

grad(can_has_svd_derivative)(np.ones((10,10)))


# In[414]:

def can_has_det_deriv(m):
    return np.linalg.det(m)


# In[416]:

np.random.seed(0)
m = np.random.randn(10,10)
np.linalg.det(m)


# In[417]:

grad(np.linalg.det)(m)


# In[481]:

def BC_w_vec(X,Y,m):
    M = np.diag(m)
    xx = np.dot(np.dot(X.T,M),X)
    xy = np.dot(np.dot(X.T,M),Y)
    yy = np.dot(np.dot(Y.T,M),Y)
    return 1-np.linalg.det(xy) / np.sqrt(np.linalg.det(xx) * np.linalg.det(yy))


# In[482]:

BC_w_vec(fs_triplets_xyz[0][0],fs_triplets_xyz[0][1],np.ones(fs_atoms))


# In[485]:

plt.plot(grad(lambda w:BC_w_vec(a,b,w))(np.ones(264)));


# In[486]:

get_ipython().magic(u'timeit grad(lambda w:BC_w_vec(a,b,w))(np.ones(264))')


# In[487]:

get_ipython().magic(u'timeit BC_w_vec(a,b,np.ones(264))')


# In[428]:

grad_bc = grad(lambda w:BC_w_vec(a,b,w))


# In[430]:

get_ipython().magic(u'timeit grad_bc(np.ones(264))')


# In[488]:

losses_unweighted = [zero_one_triplet_loss(BC_w_vec(a.xyz[0],b.xyz[0],np.ones(264)),BC_w_vec(a.xyz[0],c.xyz[0],np.ones(264))) for (a,b,c) in fs_triplets]


# In[489]:

sum(losses_unweighted) / len(losses_unweighted)


# In[490]:

def wbc_obj(weights,triplet,triplet_loss=modified_triplet_loss):
    return triplet_loss(BC_w_vec(triplet[0],triplet[1],weights),
                        BC_w_vec(triplet[0],triplet[2],weights))


# In[491]:

from msmbuilder.featurizer import RawPositionsFeaturizer as rpf


# In[492]:

fs_triplets_xyz = [(a.xyz[0],b.xyz[0],c.xyz[0]) for (a,b,c) in fs_triplets]


# In[545]:

t = time()
results = sgd(wbc_obj,fs_triplets_xyz,np.ones(fs_atoms),n_iter=100,step_size=10)
plt.plot(results[:,:10]);
print(time()-t)


# In[ ]:

results = sgd(wbc_obj,fs_triplets_xyz,np.ones(fs_atoms),n_iter=10000,step_size=1,store_intermediates=False)


# In[541]:

losses_weighted = [zero_one_triplet_loss(BC_w_vec(a.xyz[0],b.xyz[0],results[-1]),BC_w_vec(a.xyz[0],c.xyz[0],results[-1])) for (a,b,c) in fs_triplets]


# In[542]:

sum(losses_weighted) / len(losses_weighted)


# In[472]:

# idea: instead of picking the points randomly, you can pick time-points on either side of detected change-points


# In[ ]:



