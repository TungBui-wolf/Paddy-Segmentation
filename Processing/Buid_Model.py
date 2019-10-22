import numpy as np
import pandas as pd
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import BatchNormalization

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random

import matplotlib.pyplot as plt

df = np.array(pd.read_csv('Input/input_model.csv'))
print(df)

g = np.random.permutation(df)
print(g)

x_train = g[:3000,1:5]
y_train = g[:3000,5]
x_val = g[3000:4000, 1:5]
y_val = g[3000:4000, 5]

x_test = g[4000:,1:5]
y_test = g[4000:,5]

encoder =  LabelEncoder()
# print(y_train)
y1 = encoder.fit_transform(y_train)
# print(y1)
print(y1.shape)
Y_train = pd.get_dummies(y1).values
# print(Y)
print(Y_train.shape)

y2 = encoder.fit_transform(y_val)
# print(y2)
print(y2.shape)
Y_val = pd.get_dummies(y2).values
# print(Y_val)
print(Y_val.shape)

# print(y_test)
y3= encoder.fit_transform(y_test)
# print(y3)
print(y3.shape)
Y_test = pd.get_dummies(y3).values
# print(Y_test)
print(Y_test.shape)

model = Sequential()
model.add(Dense(32, input_shape=(4,)))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(2,activation='sigmoid'))

model.compile(Adam(lr=1e-4,decay=1e-7),'binary_crossentropy', ['accuracy'])
model.summary()

H = model.fit(x_train, Y_train, epochs = 100, validation_data = (x_val, Y_val))


fig = plt.figure()
numOfEpoch = 100
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='train loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['acc'], label='train accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()

scores = model.evaluate(x_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.save('Model/model_weight_paddy.h5')