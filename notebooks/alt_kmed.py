from __future__ import absolute_import, print_function, division

#from msmbuilder.cluster import MiniBatchKMedoids
from msmbuilder.cluster import *
from sklearn.utils import check_random_state
from msmbuilder import libdistance

from scipy.spatial.distance import squareform



from operator import itemgetter
import numpy as np
from sklearn.utils import check_random_state
from sklearn.base import ClusterMixin, TransformerMixin

from msmbuilder.cluster import MultiSequenceClusterMixin
from msmbuilder.cluster import _kmedoids
from msmbuilder import libdistance
from msmbuilder.base import BaseEstimator

def alt_pdist(X,metric,X_indices):
    ''' needs to just return a distance matrix for a given subset of the indices'''
    distances = np.zeros((len(X_indices),len(X_indices)))
    #print(X_indices)
    for i,ind_i in enumerate(X_indices):
        for j,ind_j in enumerate(X_indices[:i]):
            #print(len(X),len(X[0]))
            #print(ind_i)
            distances[i,j] = metric(X[ind_i],X[ind_j])
    distances = distances + distances.T
    return squareform(distances) # converts back to condensed representation

def alt_assign_nearest(X,Y,metric):
    ''' for each point in X, assign it to the nearest point in Y'''

    assignments = np.zeros(len(X))
    inertia = 0

    for i,x in enumerate(X):
        distances = np.zeros(len(Y))
        for j,y in enumerate(Y):
            distances[j] = metric(x,y)
        assignments[i] = np.argmin(distances)
        inertia += np.min(distances)**2

    return assignments,inertia

class AltMiniBatchKMedoids(MiniBatchKMedoids):

    ''' redefines fit and transform to also allow weighted rmsd instead of just 'rmsd'

    metric = {any metric accepted by minibatchkmedoids, ('callable', callable)}

    going to do this by replacing the calls to libdistance.pdist and libdistance.assign_nearest

    '''



    def fit(self, X, y=None):
        n_samples = len(X)
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        n_iter = int(self.max_iter * n_batches)
        random_state = check_random_state(self.random_state)

        cluster_ids_ = random_state.random_integers(
            low=0, high=n_samples - 1, size=self.n_clusters)
        labels_ = random_state.random_integers(
            low=0, high=self.n_clusters - 1, size=n_samples)

        n_iters_no_improvement = 0
        for kk in range(n_iter):
            # each minibatch includes the random indices AND the
            # current cluster centers
            minibatch_indices = np.concatenate([
                cluster_ids_,
                random_state.random_integers(
                    0, n_samples - 1, self.batch_size),
            ])
            X_indices=np.array(minibatch_indices, dtype=np.intp)

            if self.metric[0]=='callable':
                ''' here's where the new functionality goes! '''
                #dmat = wrmsd_dmat(X[X_indices],weights=metric[1])
                dmat = alt_pdist(X,metric=self.metric[1],X_indices=X_indices)

            else:
                dmat = libdistance.pdist(X, metric=self.metric, X_indices=X_indices)


            minibatch_labels = np.array(np.concatenate([
                np.arange(self.n_clusters),
                labels_[minibatch_indices[self.n_clusters:]]
            ]), dtype=np.intp)

            ids, intertia, _ = _kmedoids.kmedoids(
                self.n_clusters, dmat, 0, minibatch_labels,
                random_state=random_state)
            minibatch_labels, m = _kmedoids.contigify_ids(ids)

            # Copy back the new cluster_ids_ for the centers
            minibatch_cluster_ids = np.array(
                sorted(m.items(), key=itemgetter(1)))[:, 0]
            cluster_ids_ = minibatch_indices[minibatch_cluster_ids]

            # Copy back the new labels for the elements
            n_changed = np.sum(labels_[minibatch_indices] != minibatch_labels)
            if n_changed == 0:
                n_iters_no_improvement += 1
            else:
                labels_[minibatch_indices] = minibatch_labels
                n_iters_no_improvement = 0
            if n_iters_no_improvement >= self.max_no_improvement:
                break

        self.cluster_ids_ = cluster_ids_
        self.cluster_centers_ = X[cluster_ids_]

        if self.metric[0]=='callable':
            ''' here's were even more new functionality goes :) '''
            self.labels_,self.intertia_ = alt_assign_nearest(X,self.cluster_centers_,metric=self.metric[1])

        else:
            self.labels_, self.inertia_ = libdistance.assign_nearest(
                X, self.cluster_centers_, metric=self.metric)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            New data to predict.
        Returns
        -------
        Y : array, shape [n_samples,]
            Index of the closest center each sample belongs to.
        """

        if self.metric[0]=='callable':
            labels,_ = alt_assign_nearest(X,self.cluster_centers_,metric=self.metric[1])
        else:
            labels, _ = libdistance.assign_nearest(
                X, self.cluster_centers_, metric=self.metric)
        return labels
