from mdtraj.geometry import alignment
import numpy as np

def compute_atomwise_deviation_xyz(X_xyz,Y_xyz):
    X_prime = alignment.transform(X_xyz, Y_xyz)
    delta = X_prime - Y_xyz
    deviation = ((delta**2).sum(1))**0.5
    return deviation

def compute_atomwise_deviation(X,Y):
    return compute_atomwise_deviation_xyz(X.xyz[0],Y.xyz[0])

def wRMSD(X,Y,w='unweighted'):
    dev = compute_atomwise_deviation(X,Y)
    if w == 'unweighted':
        wdev = sum(dev)
    else:
        wdev = w.dot(dev)

    return np.sqrt(wdev) / len(X.xyz.T)

def compute_kinetic_weights(traj,tau=10):
    n_frames=len(traj)-tau
    n_atoms = traj.n_atoms
    atomwise_deviations = np.zeros((n_frames,n_atoms))
    for i in range(n_frames):
        atomwise_deviations[i] = compute_atomwise_deviation(traj[i],traj[i+tau])

    means = np.mean(atomwise_deviations,0)
    weights = 1/means
    return weights/sum(weights)
