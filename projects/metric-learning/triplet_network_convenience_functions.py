# generate triplets of observations from a list of trajectories, etc.


def tripletify_trajectory(X,tau_1=5,tau_2=20):
    X_triplets = []
    for i in range(len(X) - tau_2):
        X_triplets.append((X[i],X[i+tau_1],X[i+tau_2]))
    return X_triplets
