import mdtraj as md
import pyemma
import pyemma.coordinates as coor
from msmbuilder.featurizer import DihedralFeaturizer
import numpy as np

def msmbuilder_to_pyemma(msmbuilder_dih_featurizer,trajectory):
    ''' accepts an msmbuilder.featurizer.DihedralFeaturizer object + a trajectory (containing the topology
    this featurizer will be applied to) and spits out an equivalent PyEMMA featurizer '''

    all_indices = []
    for dih_type in msmbuilder_dih_featurizer.types:
        func = getattr(md, 'compute_%s' % dih_type)
        indices,_ = func(trajectory)
        all_indices.append(indices)

    indices = np.vstack(all_indices)
    sincos = msmbuilder_dih_featurizer.sincos

    pyemma_feat = coor.featurizer(trajectory.topology)
    pyemma_feat.add_dihedrals(indices,cossin=sincos)

    return pyemma_feat

if __name__=='__main__':
    # fetch some data
    from msmbuilder.example_datasets import FsPeptide
    trajectory = FsPeptide().get().trajectories[0]

    # create a DihedralFeaturizer
    from msmbuilder.featurizer import DihedralFeaturizer
    featurizer = DihedralFeaturizer(types=["phi", "psi", "chi1", "chi2"])

    # create a PyEMMA featurizer
    pyemma_feat = msmbuilder_to_pyemma(featurizer,trajectory)

    # create a tICA model
    X = pyemma_feat.transform(trajectory)
    tica = coor.tica(X,var_cutoff=0.95)

    # find the input features with the strongest correlation with
    feature_descriptors = pyemma_feat.describe()
    tica_correlations = tica.feature_TIC_correlation
    n_tics = tica_correlations.shape[1]
    best_indicator_indices = [np.argmax(np.abs(tica_correlations[:,i])) for i in range(n_tics)]
    best_indicators = [feature_descriptors[ind] for ind in best_indicator_indices]
    best_indicators_corr = [tica_correlations[best_indicator_indices[i],i] for i in range(n_tics)]

    print("These are the input features most strongly correlated with each tIC:")
    for i in range(n_tics):
        print('\ttIC{0}: {1}\n\t\twith correlation of {2:.3f}'.format(i+1,best_indicators[i],best_indicators_corr[i]))
    #print(zip(range(n_tics),best_indicators,best_indicators_corr))
