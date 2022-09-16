import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers ,regularizers
from tensorflow.keras.datasets import cifar10

physical_devices =tf.config.list_physical_devices('GPU')
#m =tf.config.experimental.set_memory_growth(physical_devices[0],True)
print("No of GPU  =",len(physical_devices))

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

# Normalize but donot use reshape for Flatten beacuse we are using CNN need to maintain the shape
x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0

# Functional API
def Soumen_model():
    inputs=tf.keras.Input(shape=(32,32,3))

    x=layers.Conv2D(32,3,padding='same',kernel_regularizer=regularizers.l2(0.01))(inputs)
    x=layers.BatchNormalization()(x)
    x=tf.keras.activations.relu(x)
    x=layers.MaxPooling2D()(x)
    
    x=layers.Conv2D(64,3,padding='same',kernel_regularizer=regularizers.l2(0.01))(inputs)
    x=layers.BatchNormalization()(x)
    x=tf.keras.activations.relu(x)

    x=layers.Conv2D(64,3,padding='same',kernel_regularizer=regularizers.l2(0.01))(inputs)
    x=layers.BatchNormalization()(x)
    x=tf.keras.activations.relu(x)

    x=layers.Flatten()(x)
    x=layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)

    x=layers.Dropout(0.5)(x)  # Drop some Connection between the upper and lower layer i.e; the "Dense" alyer and "output" layer

    outputs=layers.Dense(10)(x)
    model=keras.Model(inputs=inputs,outputs=outputs)
    return model

model=Soumen_model()
print(model.summary())

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy'],
)

model.fit(x_train,y_train,batch_size=64 ,epochs=150,verbose=2)  # when we use regularizers need to tarin more epochs=100 because we are Droping
model.evaluate(x_test,y_test,batch_size=64,verbose=2)          