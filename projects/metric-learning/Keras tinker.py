
# coding: utf-8

# In[54]:

from sklearn.datasets import load_digits
X,y = load_digits().data,load_digits().target
from sklearn.decomposition import PCA
X = PCA(20).fit_transform(X)
X.shape


# In[55]:

X_train,X_test = X[:1000],X[1000:]
y_train,y_test = Y[:1000],Y[1000:]


# In[56]:

len(set(y_test))


# In[57]:

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(20, 64, init='uniform',activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, 64, init='uniform',activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, 10, init='uniform',activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
score = model.evaluate(X_test, y_test, batch_size=16)


# In[58]:

X.shape


# In[59]:

# stolen from: https://github.com/fchollet/keras/blob/master/examples/mnist_nn.py

from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2, l1
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import numpy as np

'''
    Train a simple deep NN on the MNIST dataset.
'''

batch_size = 64
nb_classes = 10
nb_epoch = 20

np.random.seed(1337) # for reproducibility

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train=X_train.reshape(60000,784)
X_test=X_test.reshape(10000,784)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(784, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 10))
model.add(Activation('softmax'))

rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[26]:

# let's make an autoencoder

model = Sequential()
model.add(Dense(784, 128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 2,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, 128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, 784,activation='relu'))

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

model.fit(X_train, X_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=2, validation_data=(X_test, X_test))
score = model.evaluate(X_test, X_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[23]:

X_train.shape


# In[25]:

model.predict(X_train).shape


# In[33]:

model.get_weights()[:3]


# In[36]:

import copy
model_layers_archive = copy.copy(model.layers)


# In[37]:

model.layers = model_layers_archive[:6]
model.layers


# In[39]:

model.predict(X_train).shape


# In[47]:

m = Sequential()
for l in model_layers_archive[:5]:
    m.add(l)


# In[53]:

l = model_layers_archive[0]
l.get_output(X_train)


# In[48]:

model.compile(loss='mse', optimizer=rms)


# In[45]:

m.predict(X_train)


# In[ ]:

# let's do 

