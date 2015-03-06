import numpy as np
import numpy.random as npr

def transition_count_matrix(sequence):
    n = len(set(sequence)) # number of states
    C = np.zeros((n,n),dtype=int)
    for t in range(1,len(sequence)):
        i,j = sequence[t-1:t+1]
        C[i,j] += 1
    return C

def triplet_count_matrix(sequence):
    n = len(set(sequence)) # number of states
    C = np.zeros((n,n,n),dtype=int)
    for t in range(2,len(sequence)):
        i,y,j = sequence[t-2:t+1]
        C[y,i,j] += 1
    return C

def covariance(sequence):
    P_21 = transition_count_matrix(sequence)
    for i in xrange(len(P_21)):
        P_21[i] /= sum(P_21[i])
    return P_21

def trivariance(sequence):
    n = np.max(sequence)+1
    P_31 = np.zeros((n,n,n))
    for t in xrange(2,len(sequence)):
        i,x,j = sequence[t-2:t+1]
        P_31[x,i,j] += 1
    #for x in xrange(n):
    #    P_31[x] /= np.sum(P_31[x],axis=1)
    return P_31

def learn_RR_HMM(sequence,k):
    n = np.max(sequence)+1
    
    # compute empirical estimate of P_1
    P_1 = np.zeros(n)
    for i in xrange(len(sequence)):
        P_1[sequence[i]] += 1
    P_1 /= sum(P_1)
    
    # compute empirical estimate of P_21
    P_21 = covariance(sequence)
    
    # compute empirical estimates of P_3x1
    P_31 = trivariance(sequence)
    
    # compute SVD of P_21
    svd = np.linalg.svd(P_21)
    U = svd[0][:,:k]
    UT = U.T
    
    # compute model parameters
    b_1 = UT.dot(P_1)
    b_infty = np.linalg.pinv((P_21.T.dot(U))).dot(P_1)
    
    B = np.zeros((n,k,k))
    UTP21 = np.linalg.pinv(UT.dot(P_21))
    for x in xrange(n):
        B[x] = (UT.dot(P_31[x])).dot(UTP21)
    return b_1,b_infty,B