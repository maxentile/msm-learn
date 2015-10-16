import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
%matplotlib inline
import mdtraj
plt.rc('font', family='serif')

# fetch example data

from msmbuilder.example_datasets import FsPeptide
dataset = FsPeptide().get()
fs_trajectories = dataset.trajectories
fs_t = fs_trajectories[0]

# 1. internal coordinate basis sets
print('Constructing dihedral features')
from msmbuilder.featurizer import DihedralFeaturizer

basis_sets = dict()

dih_model=DihedralFeaturizer()
X = dih_model.fit_transform(fs_trajectories)
basis_sets['dihedral_phi_psi'] = X

dih_model=DihedralFeaturizer(types=['phi', 'psi', 'omega', 'chi1', 'chi2', 'chi3', 'chi4'])
X = dih_model.fit_transform(fs_trajectories)
basis_sets['dihedral_all'] = X

# 2. RMSD to reg-space refs
references = [x[::4000] for x in fs_trajectories]
refs = references[0]
for ref in references[1:]:
    refs = refs + ref
print('Computing RMSD to regularly spaced references'.format(len(refs)))

from msmbuilder.featurizer import RMSDFeaturizer
rmsdf = RMSDFeaturizer(refs)
basis_sets['rmsd_reg'] = rmsdf.fit_transform(fs_trajectories)

# 3. RMSD to cluster-center refs
print('Computing RMSD to cluster centers')

# pick cluster centers in tICA projection by k-medoids
print('  - Computing tICA projection')
from msmbuilder.decomposition import tICA
tica = tICA(lag_time=200,num_components=20)
X_tica = tica.fit_transform(X)

print('  - Finding cluster centers')
from msmbuilder.cluster import MiniBatchKMedoids
kmed = MiniBatchKMedoids(50,batch_size=200)
kmed.fit(X_tica)

# extract examplar configurations
clever_refs = []
for ind in kmed.cluster_ids_:
    clever_refs.append(fs_trajectories[ind[0]][ind[1]])

# convert list of length-1 mdtraj Trajectories to a single trajectory
# -- currently doing this in the most inefficient way possible, but it's
# a tiny list so it doesn't matter
clever_ref = clever_refs[0]
for i in range(1,len(clever_refs)):
    clever_ref = clever_ref + clever_refs[i]
clever_refs = clever_ref
print(len(clever_refs))

print('  - Computing RMSD to cluster centers')
rmsdf = RMSDFeaturizer(clever_refs)
basis_sets['rmsd_kmed'] = rmsdf.fit_transform(fs_trajectories)

rmsd_kmed_basis = np.array(basis_sets['rmsd_kmed'])

# 4. wRMSD to reg-space refs
print('Computing wRMSD to reg-space refs')
from MDAnalysis.analysis.rms import rmsd as wRMSD

# compute weights from atomwise deviations
atomwise_deviations = np.load('fs_atomwise_deviations_tau=20.npy')
mean = atomwise_deviations.mean(0)
weights = np.exp(-mean/0.065)
