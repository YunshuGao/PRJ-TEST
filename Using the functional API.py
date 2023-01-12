# Import tensorflow
import tensorflow as tf

#Define model 1 input layer shape
model1_inputs=tf.keras.Input(shape=(28*28,))

#Define model 2 input layer shape
model2_inputs=tf.keras.Input(shape=(10,))

#Define layer 1 for model 1
model1_layer1=tf.keras.layers.Dense(12,activation='relu')(model1_inputs)

#Define layer 2 for model 1
model1_layer2=tf.keras.layers.Dense(4,activation='softmax')(model1_layer1)

#Define layer1 for model 2
model2_layer1=tf.keras.layers.Dense(8,activation='relu')(model2_inputs)

#Define layer2 for model 2
model2_layer2=tf.keras.layers.Dense(4,activation='softmax')(model2_layer1)

#Merge model 1 and model 2
merged=tf.keras.layers.add([model1_layer2,model2_layer2])

#Define a functional model
model=tf.keras.Model(inputs=[model1_inputs,model2_inputs],outputs=merged)

#Compile the model
model.compile('adam',loss='categorical_crossentropy')

print(model.summary())