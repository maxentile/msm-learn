'''
trying to come up with alternatives to tICA for deriving
collective variables that are "slow projections"

approach: use autograd so I can test out different loss
functions easily
'''

from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from msmbuilder.decomposition import tICA

def autocorr(X,tau=1):
    mu = X.mean(0)
    X_ = X-mu
    M = len(X) - tau
    dim = len(X.T)
    corr = np.zeros((dim,dim))
    for i in range(M):
        corr += np.outer(X_[i],X_[i+tau]) + np.outer(X_[i+tau],X_[i])
    return corr / (2.0*M)

def autocorr_strength(corr_matrix):
    # what makes sense here, sum of absolute values?
    # sum of squared values? maximum value?
    #return np.sum(corr_matrix)

    #return np.sum(np.abs(corr_matrix))
    #return np.sum(corr_matrix**2)
    #return np.sum(np.diag(corr_matrix)**2)
    return np.sum(np.abs(np.diag(corr_matrix)))

# stolen from: https://github.com/HIPS/autograd/blob/master/examples/optimizers.py
def rmsprop(grad, x, num_iters=100, step_size=1.0, gamma=0.9, eps = 10**-8):
    """Root mean squared prop: See Adagrad paper for details."""
    avg_sq_grad = np.ones(len(x))
    for i in xrange(num_iters):
        g = grad(x)
        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x -= step_size * g/(np.sqrt(avg_sq_grad) + eps)
    return x

# defining an objective function that seeks to maximize
# autocorrelation strength
class SlowProjections():
    def __init__(self,n_components=2,tau=1):
        self.n_components = n_components
        self.tau=tau

    def objective(self,X,A,gamma=0.1):
        X_ = np.dot(X,A)
        ac = autocorr(X_,self.tau)
        penalty = gamma*np.sum(np.abs(A))
        return autocorr_strength(ac) + penalty

    def generate_obj(self,X):
        return lambda A:self.objective(self,X,A)

    def fit(self,X,n_iter=100,A_init=None):
        if A_init==None:

            pca = PCA(self.n_components)
            pca.fit(X)
            A_init = pca.components_.T
            A_init = np.reshape(A_init,np.prod(A_init.shape))
            #A_init = np.ones((X.shape[1],self.n_components))
        #def obj(self,A):
        #    return self.objective(X,A_)
        def obj(A):
            A_ = np.reshape(A,(X.shape[1],self.n_components))
            X_ = np.dot(X,A_)
            X_ /= np.std(X_,0)
            X_ -= np.mean(X_,0)
            ac = autocorr(X_,self.tau)
            return -autocorr_strength(ac)

        #obj = lambda A: self.objective(X,A)
        obj_grad = grad(obj)
        #obj_grad = grad(lambda self,A: self.objective(X,A))
        #obj_grad = grad(self.generate_obj(X))
        A_final = rmsprop(obj_grad,A_init,n_iter)
        self.A = np.reshape(A_final,(X.shape[1],self.n_components))
        #return A_final

    def transform(self,X):
        return np.dot(X,self.A)

    def fit_transform(self,X,n_iter=100,A_init=None):
        self.fit(X,n_iter,A_init)
        return self.transform(X)

if __name__=='__main__':
    X = np.random.randn(10000,10)

    i
    FsPeptide=True
    if FsPeptide:
        from msmbuilder.example_datasets import AlanineDipeptide,FsPeptide
        dataset = FsPeptide().get()
        fs_trajectories = dataset.trajectories
        from msmbuilder import featurizer
        dhf = featurizer.DihedralFeaturizer()
        dhft = dhf.fit_transform(fs_trajectories)
        X = dhft[0]
    #X_dihedral = np.vstack(dhft)
    sp =  SlowProjections()
    X_sp = sp.fit_transform(X)

    pca = PCA(2)
    X_pca = pca.fit_transform(X)

    tica = tICA(2,10)
    X_tica = tica.fit_transform([X])[0]

    results = [('Slow projections',X_sp),
               ('PCA',X_pca),
               ('tICA',X_tica)]
    for name,X_ in results:
        plt.scatter(X_[:,0],X_[:,1],linewidths=0,s=4,
            c=np.arange(len(X_)),alpha=0.5)
        X_ /= np.std(X_,0)
        X_ -= np.mean(X_,0)
        perf = autocorr_strength(autocorr(X_))
        title = name + ': {0:.3f}'.format(perf)
        plt.title(title)
        plt.savefig(name + '.jpg',dpi=300)
        plt.close()

    print(sp.A)
    print(np.sum(sp.A==0))
