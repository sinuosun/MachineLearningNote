import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units = 32 , activation = 'sigmoid'),
    Dense(units = 32 , activation = 'sigmoid'),
    Dense(units = 16 , activation = 'sigmoid'),
    Dense(units = 8 , activation = 'sigmoid'),
    Dense(units = 1 , activation = 'sigmoid'),
])

import numpy as np

train_data = np.loadtxt("trainDL_v1.txt")

X = train_data[:, :2]
Y = train_data[:, 2]

from tensorflow.keras.losses import BinaryCrossentropy
model.compile(loss = BinaryCrossentropy())
model.fit(X,Y,epochs = 500, verbose = 0)

test_data = np.loadtxt("testDL_v1.txt")

X_test = test_data[:, :2]
Y_test = test_data[:, 2]
pred = model.predict(X_test,verbose = 0)
print(pred)