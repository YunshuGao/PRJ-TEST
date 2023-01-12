# Import tensorflow
from tensorflow import keras

#Define a sequential model
model=keras.Sequential()

#Define first hidden layer
model.add(keras.layers.Dense(16,activation='relu',input_shape=(28*28,)))

#Define second hidden layer
model.add(keras.layers.Dense(8,activation='relu'))

#Define output layer
model.add(keras.layers.Dense(4,activation='softmax'))

#Compile the model
model.compile('adam',loss='categorical_crossentropy')

#summarize the model
print(model.summary())