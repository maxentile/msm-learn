from mdtraj.geometry import alignment
import numpy as np

def compute_atomwise_deviation_xyz(X_xyz,Y_xyz):
    ''' given two sets of coordinates as numpy arrays,
    align them and return the vector of distances between
    corresponding pairs of atoms'''
    X_prime = alignment.transform(X_xyz, Y_xyz)
    delta = X_prime - Y_xyz
    deviation = ((delta**2).sum(1))**0.5
    return deviation

def compute_atomwise_deviation(X,Y):
    ''' given trajectory frames, compute atomwise deviations'''
    return compute_atomwise_deviation_xyz(X.xyz[0],Y.xyz[0])

def wRMSD(X,Y,w='unweighted'):
    ''' compute weighted RMSD using a weight vector'''
    dev = compute_atomwise_deviation(X,Y)
    if w == 'unweighted':
        wdev = sum(dev)
    else:
        wdev = w.dot(dev)

    return np.sqrt(wdev) / len(X.xyz.T)

def wRMSD_xyz(X,Y,w='unweighted'):
    ''' compute weighted RMSD using a weight vector'''
    dev = compute_atomwise_deviation_xyz(X,Y)
    if w == 'unweighted':
        wdev = sum(dev)
    else:
        wdev = w.dot(dev)

    return np.sqrt(wdev) / len(X.T)

def compute_kinetic_weights(traj,tau=10):
    ''' for all tau-lagged pairs of observations in traj, compute their
    atomwise_deviations. Return (the mean over the observation pairs)**-1'''
    n_frames=len(traj)-tau
    n_atoms = traj.n_atoms
    atomwise_deviations = np.zeros((n_frames,n_atoms))
    for i in range(n_frames):
        atomwise_deviations[i] = compute_atomwise_deviation(traj[i],traj[i+tau])

    means = np.mean(atomwise_deviations,0)
    weights = 1/means
    return weights/sum(weights)

def sgd(objective,dataset,init_point,batch_size=20,n_iter=100,step_size=0.01,seed=0):
    ''' objective takes in a parameter vector and an array of data'''
    np.random.seed(seed)
    testpoints = np.zeros((n_iter,len(init_point)))
    testpoints[0] = init_point
    shuffled = np.array(dataset)
    np.random.shuffle(shuffled)
    accept_frac = 1.0*batch_size/dataset.shape[0]
    ind=0
    for i in range(1,n_iter):
        #max_ind = ind+batch_size
        #if max_ind<len(dataset):
        #    subset = dataset[ind:max_ind]
        #    ind = (ind + batch_size)
        #else:
        #    new_ind = (max_ind-len(dataset))
        #    subset = np.vstack([dataset[ind:],dataset[:new_ind]])
        subset = dataset[np.random.rand(len(dataset))<accept_frac]
        obj_grad = grad(lambda p:objective(p,subset))
        gradient = np.nan_to_num(obj_grad(testpoints[i-1]))
        #print(gradient)
        testpoints[i] = testpoints[i-1] - gradient*step_size

    return testpoints

def near_far_triplet_loss(metric,batch,tau_1=1,tau_2=10):
    ''' batch is a numpy array of time-ordered observations'''
    #triplets = np.array([(batch[i],batch[i+tau_1],batch[i+tau_2]) for i in range(len(batch)-tau_2)])
    cost=0
    n_triplets = len(batch)-tau_2
    for i in range(n_triplets):
        cost+=penalty(metric,batch[i],batch[i+tau_1],batch[i+tau_2])
    return cost / n_triplets

def triplet_wrmsd_loss(weights,dataset,tau_1=1,tau_2=10):
    ''' given a weight vector and a dataset, compute the
    '''
    metric = lambda x,y:wRMSD(x,y,weights)
    return near_far_triplet_loss(metric,dataset,tau_1,tau_2)
