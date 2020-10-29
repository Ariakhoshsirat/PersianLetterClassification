import random

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, Conv2D, SeparableConv2D, Reshape, \
    SimpleRNN, LocallyConnected2D, Embedding, LSTM, MaxPooling1D
from keras.models import Model, Sequential
import keras
import numpy as np
import matplotlib.pyplot as plt
import theano
import matplotlib.pyplot as plt
import theano.tensor as T
import pandas as pd
from keras.optimizers import RMSprop, SGD

TrainX = pd.read_csv('X_train' , header = None)
TrainY = pd.read_csv('TrainLabels' , header = None)


TestX = pd.read_csv('X_test' , header = None)
TestY = pd.read_csv('TestLabels' , header = None)


print(TrainY.shape)

TrainY = keras.utils.to_categorical(TrainY-1)
TestY = keras.utils.to_categorical(TestY-1)

Y_train = TrainY

Y_test = TestY

# print(TrainY.shape)

# TrainY = pd.get_dummies(TrainY)
# Y_train = Y_train.transpose()
print(Y_train.shape)


# print('Fin.')

TrainX = TrainX.astype('float32')
TrainX = TrainX.as_matrix()

TestX = TestX.astype('float32')
TestX = TestX.as_matrix()


X_train = TrainX #.reshape(TrainX.shape[0],95, 95, 1)
X_test = TestX #.reshape(TestX.shape[0],95, 95, 1)

X_train /= 255
X_test /= 255

model = Sequential()

# model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(95, 95 , 1)))
# model.add(Conv2D(32, (5, 5), activation='relu'))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.3))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(17, activation='softmax'))


model.add(Dense(512, activation='relu',input_shape=(9025,)))
# model.add(Dropout(0.1))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(17, activation='softmax'))


model.summary()


print('Network Constructed')

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=128, epochs=30, verbose=1)


score = model.predict(X_train, verbose=1)
acc = 0;
for i in range(score.shape[0]):
    if (score[i,:].argmax() == Y_train[i,:].argmax()):
        acc += 1;
accuracy = acc / score.shape[0]
print('\n')
print(accuracy)


score = model.predict(X_test, verbose=1)

acc = 0;
for i in range(score.shape[0]):
    if (score[i,:].argmax() == Y_test[i,:].argmax()):
        acc += 1;
accuracy = acc / score.shape[0]
print('\n')
print(accuracy)

#-------------------    10% Noise    -----------------------

X_test1 =  TestX
indsToNoise = random.sample(range(1, 9025), np.ceil(9025 * 1 / 10).astype(int))

print('s')
for i in range(17):
    X_test1[i,indsToNoise] *= -1

# X_test1 = X_test1.reshape(TestX.shape[0],95, 95, 1)

score = model.predict(X_test1, verbose=1)

acc = 0;
for i in range(score.shape[0]):
    if (score[i,:].argmax() == Y_test[i,:].argmax()):
        acc += 1;
accuracy = acc / score.shape[0]
print('\n')
print(accuracy)

#-------------------    20% Noise    -----------------------
X_test2 =  TestX
indsToNoise = random.sample(range(1, 9025), np.ceil(9025 * 2 / 10).astype(int))

print('s')
for i in range(17):
    X_test2[i,indsToNoise] *= -1


# X_test2 = X_test2.reshape(TestX.shape[0],95, 95, 1)
score = model.predict(X_test2, verbose=1)

acc = 0;
for i in range(score.shape[0]):
    if (score[i,:].argmax() == Y_test[i,:].argmax()):
        acc += 1;
accuracy = acc / score.shape[0]
print('\n')
print(accuracy)

#-------------------    30% Noise    -----------------------
X_test3 =  TestX
indsToNoise = random.sample(range(1, 9025), np.ceil(9025 * 3 / 10).astype(int))

print('s')
for i in range(17):
    X_test3[i,indsToNoise] *= -1

# X_test3 = X_test3.reshape(TestX.shape[0],95, 95, 1)

score = model.predict(X_test3, verbose=1)

acc = 0;
for i in range(score.shape[0]):
    if (score[i,:].argmax() == Y_test[i,:].argmax()):
        acc += 1;
accuracy = acc / score.shape[0]
print('\n')
print(accuracy)


print('Fin.')




