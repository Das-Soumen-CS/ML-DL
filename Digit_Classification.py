import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices =tf.config.list_physical_devices('GPU')
#m =tf.config.experimental.set_memory_growth(physical_devices[0],True)
print("No of GPU  =",physical_devices)


physical_devices =tf.config.list_physical_devices('CPU')
print("No of CPU =",physical_devices,"\n")

(x_train,y_train),(x_test,y_test)=mnist.load_data()

print("The shape of x_train =",x_train.shape)
print("The shape of y_train =",y_train.shape)

print("The shape of x_test =",x_test.shape)
print("The shape of y_test =",y_test.shape,"\n")

# Normalization
x_train=x_train.reshape(-1,28*28).astype("float32")/255.0
print("Normalized X_train=",x_train,"\n")

x_test=x_test.reshape(-1,28*28).astype("float32")/255.0
print("Normalized X_test=",x_test,"\n")

# Create Sequetial API (Not flexible like Functional API)
model=keras.Sequential(
    [
        layers.Dense(256,activation='relu'),
        layers.Dense(512,activation='relu'),
        layers.Dense(10),

    ]
)

# we can add one layer at a time also same as above but one layer added at a time
model=keras.Sequential()
model.add(keras.Input(shape=(28*28)))
model.add(layers.Dense(256,activation='relu'))
#print(model.summary(),"\n")
model.add(layers.Dense(512,activation='relu'))
#print(model.summary(),"\n")
model.add(layers.Dense(10))
#print(model.summary())

# Functional API

inputs = keras.Input(shape=(28*28))
temp=layers.Dense(256,activation='relu',name='First_Layer')(inputs)
temp=layers.Dense(512,activation='relu',name="Second_layer")(temp)
outputs=layers.Dense(10,activation='softmax',name='Output_layer')(temp)  # when use "softmax" must change from_logits=False
model=keras.Model(inputs=inputs,outputs=outputs)


 # For Compiling
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

print(model.summary())

# fit the model to complie
model.fit(x_train,y_train,batch_size=10,epochs=5,verbose=2)

#print(model.summary())
model.evaluate(x_test,y_test,batch_size=10,verbose=2)