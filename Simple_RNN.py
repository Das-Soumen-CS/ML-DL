import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices =tf.config.list_physical_devices('GPU')
#m =tf.config.experimental.set_memory_growth(physical_devices[0],True)
print("No of GPU  =",len(physical_devices))

(x_train,y_train),(x_test,y_test)=mnist.load_data()

# Normalize but donot use reshape for Flatten beacuse we are using CNN need to maintain the shape
x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0

model=tf.keras.Sequential()
model.add(keras.Input(shape=(None,28)))

model.add(
    layers.SimpleRNN(512,return_sequences=True,activation='relu',name='First_Layer')
)

model.add(layers.SimpleRNN(512,activation='relu',name='Second_Layer'))
model.add(layers.Dense(10,name='output_Layer'))

print(model.summary())

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),   # from_logits=True  beacuse in our model don't have any softmax activation function to the last "Dense" layer
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy'],
)

model.fit(x_train,y_train,batch_size=64 ,epochs=10,verbose=2)
model.evaluate(x_test,y_test,batch_size=64,verbose=2)
       