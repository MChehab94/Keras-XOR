from keras import Sequential
from keras.layers import Dense
import numpy as np

xor = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.asarray([[0], [1], [1], [1]])


model = Sequential(layers=[
    Dense(units=2, input_shape=(2,), activation='tanh', name='input'),
    Dense(units=1, activation='sigmoid', name='output')
])

model.compile(optimizer='sgd', metrics=['accuracy'], loss='binary_crossentropy')
model.fit(xor, y_xor, epochs=1000)

print("predicting [1, 0]: ")
print(model.predict_classes(np.asarray([[1, 0]])))
print("Predicting [0, 1]:")
print(model.predict_classes(np.asarray([[0, 1]])))
print("Predicting [0, 0]:")
print(model.predict_classes(np.asarray([[0, 0]])))
print("Predicting [1, 1]:")
print(model.predict_classes(np.asarray([[1, 1]])))