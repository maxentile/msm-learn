# goal: find an embedding based on RMSD (dis)similarities

import numpy as np
import pylab as pl
import numpy.random as npr

pl.rcParams['image.cmap'] = 'gray'

# import data
from msmbuilder.example_datasets import AlanineDipeptide
dataset = AlanineDipeptide().get()
ala_trajectories = dataset.trajectories

from msmbuilder.example_datasets import FsPeptide
dataset = FsPeptide().get()
fs_trajectories = dataset.trajectories

t = ala_trajectories[0]
fs_t = fs_trajectories[0]

# compute pairwise RMSD among snapshots of the chosen trajectory 
import mdtraj

n = 2000
rmsd_fs = np.zeros((n,n))
rmsd_ala = np.zeros((n,n))

for i in range(1,n):
    rmsd_ala[i,:i] = mdtraj.rmsd(t[:i],t[i])
    rmsd_fs[i,:i] = mdtraj.rmsd(fs_t[:i],fs_t[i])

rmsd_ala = (rmsd_ala + rmsd_ala.T)[1:,1:]
rmsd_fs = (rmsd_fs + rmsd_fs.T)[1:,1:]

# plot figures
pl.rc('font', family='serif')
pl.imshow(rmsd_ala,interpolation='none')
pl.colorbar()
pl.title('Pairwise RMSD: Alanine dipeptide')
pl.savefig('figures/alanine-dipeptide-rmsd-gray.jpg',dpi=300)
pl.close()

pl.imshow(rmsd_fs,interpolation='none')
pl.colorbar()
pl.title('Pairwise RMSD: Fs peptide')
pl.savefig('figures/fs-peptide-rmsd-gray.jpg',dpi=300)
pl.close()