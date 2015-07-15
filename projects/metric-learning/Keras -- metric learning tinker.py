
# coding: utf-8

# In[1]:

import numpy as np

from msmbuilder.example_datasets import FsPeptide
fs = FsPeptide().get()

from msmbuilder.featurizer import DihedralFeaturizer
dhf = DihedralFeaturizer()
dhft = dhf.fit_transform(fs.trajectories)

from sklearn.decomposition import PCA
pca = PCA(whiten=True)
pca.fit(np.vstack(dhft))

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.plot(100*np.cumsum(pca.explained_variance_ratio_))
plt.hlines(100.0,0,pca.n_components_,linestyles='--')
plt.ylim(0,100)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance (%)')

#n_comp = sum(np.cumsum(pca.explained_variance_ratio_)<0.95)

#X_ = pca.transform(np.vstack(dhft))[:,:n_comp]

X_ = np.vstack(dhft)
X_ -= X_.mean(0)
X_ /= X_.std(0)
X_train_ = X_[:200000]
X_test_ = X_[200000:]

#npr.seed(0)
#mask = npr.rand(len(X_))<0.7
#X_train_ = X_[mask]
#X_test_ = X_[-mask]


# In[53]:

print('The first 2 components only explain {0:.2f}% of the variance'.format(100*np.cumsum(pca.explained_variance_ratio_)[1]))


# In[3]:

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb


# In[12]:

from keras.layers.recurrent import LSTM

lstm= LSTM(10, output_dim=128, 
        init='glorot_uniform', inner_init='orthogonal', 
        activation='tanh', inner_activation='hard_sigmoid',
        weights=None, truncate_gradient=-1, return_sequences=False)


# In[14]:

model = Sequential()
model.add(LSTM(256, 128)) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))


# In[16]:

imdb_data = imdb.load_data()


# In[22]:

X_[0].shape,np.hstack(X_[:2]).shape


# In[7]:

sequence.skipgrams(X_[:100],10)


# In[94]:

def generate_skip_gram_couples(sequences,neighborhood_size=10,n_examples=100000,prop_positive=0.5):
    np.random.seed(0)
    couples = np.zeros((n_examples,2*sequences[0].shape[1]))
    y = np.zeros((n_examples,2))
    
    for i in range(n_examples):
        ind1=np.random.randint(len(sequences))
        sequence = sequences[ind1]
        pivot = np.random.randint(len(sequence)-neighborhood_size)
        if np.random.rand()<prop_positive:
            label=1
            other = np.random.randint(neighborhood_size)+pivot
            couples[i] = np.hstack((sequence[pivot],sequence[other]))
        else:
            label=0
            ind2 = np.random.randint(len(sequences))
            sequence2 = sequences[ind2]
            other = np.random.randint(len(sequence))
            while ind1==ind2 and abs(other-pivot) < neighborhood_size:
                ind2 = np.random.randint(len(sequences))
                sequence2 = sequences[ind2]
                other = np.random.randint(len(sequence2))
            couples[i] = np.hstack((sequence[pivot],sequence2[other]))
                
        
        y[i,label] = 1
    return couples,y


# In[95]:

X,y=generate_skip_gram_couples(dhft)


# In[96]:

np.random.seed(0)
mask = np.random.rand(len(X))<0.7
X_train,y_train = X[mask],y[mask]
X_test,y_test = X[-mask],y[-mask]


# In[97]:

X.shape,y.shape


# In[98]:

model = Sequential()
model.add(Dense(168, 200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200, 50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50, 2))
model.add(Activation('relu'))
model.add(Dense(2, 2))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

model.fit(X_train, y_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[99]:

model = Sequential()
model.add(Dense(168, 200))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(200, 50))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(50, 2))
model.add(Activation('tanh'))
model.add(Dense(2, 2))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

model.fit(X_train, y_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[100]:

model = Sequential()
model.add(Dense(168, 200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200, 50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50, 2))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)


# In[101]:

model.fit(X_train, y_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[102]:

model = Sequential()
model.add(Dense(168, 2))
model.add(Activation('linear'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

model.fit(X_train, y_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[103]:

model = Sequential()
model.add(Dense(168, 200))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(200, 50))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(50, 2))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[104]:

model = Sequential()
model.add(Dense(168, 200))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(200, 50))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(50, 2))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[105]:

model = Sequential()
model.add(Dense(168, 200))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(200, 5))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(5, 2))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[106]:

model = Sequential()
model.add(Dense(168, 100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(100, 2))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2, 2))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[107]:

model = Sequential()
model.add(Dense(168, 2))
model.add(Activation('relu'))
model.add(Dropout(0.2))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[108]:

model = Sequential()
model.add(Dense(168, 300))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(300, 2))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2, 2))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[109]:

def generate_kinetic_distance_pairs(sequences,max_kinetic_distance=1000,n_examples=100000):
    np.random.seed(0)
    pairs = np.zeros((n_examples,2*sequences[0].shape[1]))
    y = np.zeros((n_examples))
    
    for i in range(n_examples):
        sequence = sequences[np.random.randint(len(sequences))]
        pivot = np.random.randint(len(sequence)-max_kinetic_distance)
        kinetic_distance = np.random.randint(1,max_kinetic_distance)
        other = pivot + kinetic_distance
        
        pairs[i] = np.hstack((sequence[pivot],sequence[other]))
        y[i] = kinetic_distance
    y -= y.mean()
    y /= y.std()
    return pairs,y


# In[110]:

X,y = generate_kinetic_distance_pairs(dhft)
np.random.seed(0)
mask = np.random.rand(len(X))<0.7
X_train,y_train = X[mask],y[mask]
X_test,y_test = X[-mask],y[-mask]


# In[111]:

model = Sequential()
model.add(Dense(168, 1))
model.add(Activation('linear'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=20, nb_epoch=20, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[113]:

y_pred = model.predict(X_test)
plt.hist(y_pred,bins=50);


# In[116]:

plt.scatter(y_pred,y_test,linewidths=0,alpha=0.5,s=2)
plt.xlabel('linear prediction')
plt.ylabel('actual')
plt.title('kinetic distance')


# In[72]:

model = Sequential()
model.add(Dense(168, 300))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(300, 20))
model.add(Activation('relu'))
model.add(Dense(20, 1))
model.add(Activation('relu'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=20, nb_epoch=20, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[75]:

model = Sequential()
model.add(Dense(168, 100))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(100, 10))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(10, 1))
model.add(Activation('tanh'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=20, nb_epoch=20, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[76]:

model = Sequential()
model.add(Dense(168, 200))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(200, 20))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(20, 1))
model.add(Activation('tanh'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=20, nb_epoch=20, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[125]:

model = Sequential()
model.add(Dense(168, 168))
model.add(Activation('tanh'))
model.add(Dense(168, 500))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(500, 50))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(50, 1))
model.add(Activation('tanh'))

rms = RMSprop()
model.compile(loss='mae', optimizer=rms)
model.fit(X_train, y_train, batch_size=50, nb_epoch=100, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[126]:

y_pred = model.predict(X_test)
plt.hist(y_pred,bins=50);

plt.figure()

plt.scatter(y_pred,y_test,linewidths=0,alpha=0.5,s=2)
plt.xlabel('nonlinear prediction (deep tanh net)')
plt.ylabel('actual')
plt.title('kinetic distance')


# In[132]:

model_relu = Sequential()
model_relu.add(Dense(168, 300))
model_relu.add(Activation('relu'))
model_relu.add(Dropout(0.5))
model_relu.add(Dense(300, 100))
model_relu.add(Activation('relu'))
model_relu.add(Dropout(0.5))
model_relu.add(Dense(100, 50))
model_relu.add(Activation('relu'))
model_relu.add(Dropout(0.5))
model_relu.add(Dense(50, 1))

rms = RMSprop()
model_relu.compile(loss='mae', optimizer=rms)
model_relu.fit(X_train, y_train, batch_size=50, nb_epoch=1000, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model_relu.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[133]:

y_pred = model_relu.predict(X_test)
plt.hist(y_pred,bins=50);

plt.figure()

plt.scatter(y_pred,y_test,linewidths=0,alpha=0.5,s=2)
plt.xlabel('nonlinear prediction (deep relu net)')
plt.ylabel('actual')
plt.title('kinetic distance')


# In[134]:

model_relu = Sequential()
model_relu.add(Dense(168, 500))
model_relu.add(Activation('relu'))
model_relu.add(Dropout(0.5))
model_relu.add(Dense(500, 200))
model_relu.add(Activation('relu'))
model_relu.add(Dropout(0.5))
model_relu.add(Dense(200, 50))
model_relu.add(Activation('relu'))
model_relu.add(Dropout(0.5))
model_relu.add(Dense(50, 10))
model_relu.add(Activation('relu'))
model_relu.add(Dense(10, 1))

rms = RMSprop()
model_relu.compile(loss='mae', optimizer=rms)
model_relu.fit(X_train, y_train, batch_size=50, nb_epoch=1000, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model_relu.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[137]:

plt.hist(y_test,bins=50);
plt.figure('distribution of distances')

y_pred = model_relu.predict(X_test)
plt.hist(y_pred,bins=50);
plt.figure('distribution of predictions')

plt.figure()

plt.scatter(y_pred,y_test,linewidths=0,alpha=0.5,s=2)
plt.xlabel('nonlinear prediction (deep relu net)')
plt.ylabel('actual')
plt.title('kinetic distance')


# In[80]:

model = Sequential()
model.add(Dense(168, 50))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(50, 1))
model.add(Activation('relu'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=10, nb_epoch=200, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[ ]:

# and how well can we do with just the aligned atomic distances?


# In[140]:

from mdtraj.geometry import alignment

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


# In[143]:

fs.trajectories[0].n_atoms


# In[150]:

def generate_skip_gram_deviations(sequences,neighborhood_size=10,n_examples=100000,prop_positive=0.5):
    np.random.seed(0)
    deviations = np.zeros((n_examples,sequences[0].n_atoms))
    y = np.zeros((n_examples,2))
    
    for i in range(n_examples):
        ind1=np.random.randint(len(sequences))
        sequence = sequences[ind1]
        pivot = np.random.randint(len(sequence)-neighborhood_size)
        if np.random.rand()<prop_positive:
            label=1
            other = np.random.randint(neighborhood_size)+pivot
            deviations[i] = compute_atomwise_deviation(sequence[pivot],sequence[other])
        else:
            label=0
            ind2 = np.random.randint(len(sequences))
            sequence2 = sequences[ind2]
            other = np.random.randint(len(sequence))
            while ind1==ind2 and abs(other-pivot) < neighborhood_size:
                ind2 = np.random.randint(len(sequences))
                sequence2 = sequences[ind2]
                other = np.random.randint(len(sequence2))
            deviations[i] = compute_atomwise_deviation(sequence[pivot],sequence2[other])
        
        if i % (n_examples / 50) == 0:
            print(i)
        
        y[i,label] = 1
    return deviations,y


# In[155]:

X_deviations,y_dev = generate_skip_gram_deviations(fs.trajectories,n_examples=100000)


# In[156]:

np.random.seed(0)
mask = np.random.rand(len(X_deviations))<0.7
X_train,y_train = X_deviations[mask],y_dev[mask]
X_test,y_test = X_deviations[-mask],y_dev[-mask]


# In[157]:

n_atoms = len(X_train.T)


# In[158]:

model = Sequential()
model.add(Dense(n_atoms, 100))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(100, 2))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2, 2))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[159]:

model = Sequential()
model.add(Dense(n_atoms, 100))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(100, 2))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(2, 2))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[215]:

model = Sequential()
model.add(Dense(n_atoms, 2))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, y_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, show_accuracy=False, verbose=0)


# In[216]:

layer = model.layers[0]
weights = layer.W.get_value()


# In[218]:

labels = (y_test*np.vstack((np.zeros(len(y_test)),np.ones(len(y_test)))).T).sum(1)


# In[225]:

X_unsup = PCA(2).fit_transform(X_test)


# In[232]:

plt.scatter(X_unsup[:,0],X_unsup[:,1],s=1,alpha=0.1,linewidths=0,c=labels)


# In[235]:

from sklearn.lda import LDA
lda = LDA(2)
X_sup = lda.fit_transform(X_test,labels)
X_sup.shape


# In[236]:

plt.scatter(X_sup,np.random.randn(len(X_sup)),s=1,alpha=0.1,linewidths=0,c=labels)


# In[217]:

X_pred_lin = PCA(2).fit_transform(np.dot(X_test,weights))


# In[219]:

plt.scatter(X_pred_lin[:,0],X_pred_lin[:,1],s=1,alpha=0.1,linewidths=0,c=labels)


# In[223]:

plt.plot(weights[:,0])


# In[224]:

plt.scatter(weights[:,0],weights[:,1])


# In[239]:

from msmbuilder.featurizer import RawPositionsFeaturizer
rpft = RawPositionsFeaturizer().fit_transform(fs.trajectories)


# In[245]:

X = np.vstack(rpft)


# In[247]:

X.shape


# In[248]:

np.random.seed(0)
mask = np.random.rand(len(X))<0.7
X_train = X[mask]
X_test = X[-mask]


# In[250]:

n_coords = X.shape[1]
model = Sequential()
model.add(Dense(n_coords, 2))
model.add(Dense(2, n_coords))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, X_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, X_test))
score = model.evaluate(X_test, X_test, show_accuracy=False, verbose=0)


# In[254]:

layer = model.layers[0]
weights = layer.W.get_value()


# In[251]:

X_pca = PCA(2).fit_transform(X)


# In[252]:

plt.scatter(X_pca[:,0],X_pca[:,1],linewidths=0,alpha=0.5,s=2)


# In[255]:

X_nn_pca = np.dot(X,weights)


# In[256]:

plt.scatter(X_nn_pca[:,0],X_nn_pca[:,1],linewidths=0,alpha=0.5,s=2)


# In[ ]:

n_coords = X.shape[1]
model = Sequential()
model.add(Dense(n_coords, 2))

model.add(Dense(2, n_coords))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, X_train, batch_size=10, nb_epoch=10, show_accuracy=True, verbose=2, validation_data=(X_test, X_test))
score = model.evaluate(X_test, X_test, show_accuracy=False, verbose=0)


# In[258]:

n_coords = X.shape[1]
model = Sequential()
model.add(Dense(n_coords, 50))
model.add(Activation('tanh'))
model.add(Dense(50, 2))
model.add(Activation('tanh'))
model.add(Dense(2, 50))
model.add(Activation('tanh'))
model.add(Dense(50, n_coords))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, X_train, batch_size=10, nb_epoch=100, show_accuracy=True, verbose=2, validation_data=(X_test, X_test))
score = model.evaluate(X_test, X_test, show_accuracy=False, verbose=0)


# In[277]:

layer1 = model.layers[0]
layer1_out = np.tanh(np.dot(X,layer1.W.get_value())+layer1.b.get_value())
layer2 = model.layers[2]
layer2_out = layer2.activation(np.dot(layer1_out,layer2.W.get_value())+layer2.b.get_value())
layer1_out.shape,layer2_out.shape


# In[278]:

X_nn = layer2_out


# In[280]:

plt.scatter(X_nn[:,0],X_nn[:,1],linewidths=0,alpha=0.5,s=2,c=np.arange(len(X_nn)))


# In[4]:

# now with dihedral angles instead

X = np.vstack(dhft)
np.random.seed(0)
mask = np.random.rand(len(X))<0.7
X_train = X[mask]
X_test = X[-mask]

n_coords = X.shape[1]
model = Sequential()
model.add(Dense(n_coords, 50))
model.add(Activation('tanh'))
model.add(Dense(50, 2))
model.add(Activation('tanh'))
model.add(Dense(2, 50))
model.add(Activation('tanh'))
model.add(Dense(50, n_coords))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)
model.fit(X_train, X_train, batch_size=10, nb_epoch=100, show_accuracy=True, verbose=2, validation_data=(X_test, X_test))
score = model.evaluate(X_test, X_test, show_accuracy=False, verbose=0)


# In[289]:

def partial_apply(model,X):
    layer1 = model.layers[0]
    layer1_out = np.tanh(np.dot(X,layer1.W.get_value())+layer1.b.get_value())
    layer2 = model.layers[2]
    layer2_out = layer2.activation(np.dot(layer1_out,layer2.W.get_value())+layer2.b.get_value())
    return layer2_out


# In[291]:

X_nn = partial_apply(model,X)
plt.scatter(X_nn[:,0],X_nn[:,1],linewidths=0,alpha=0.5,s=2,c=np.arange(len(X_nn)))


# In[2]:

X_pca = PCA(2).fit_transform(np.vstack(dhft))
plt.scatter(X_pca[:,0],X_pca[:,1],linewidths=0,alpha=0.5,s=2,c=np.arange(len(X_pca)))


# In[287]:

pca = PCA(2)
pca.fit(X)
np.sum((pca.inverse_transform(pca.transform(X))-X)**2)/(len(X)*X.shape[1])


# In[286]:

X.shape


# In[282]:

dhft[0].shape


# In[81]:

from msmbuilder.example_datasets import MetEnkephalin


# In[82]:

met = MetEnkephalin().get()


# In[84]:

print(met.DESCR)


# In[ ]:



