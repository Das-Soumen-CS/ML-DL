import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
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

    x=layers.Conv2D(32,3)(inputs)
    x=layers.BatchNormalization()(x)
    x=tf.keras.activations.relu(x)
    x=layers.MaxPooling2D()(x)
    
    x=layers.Conv2D(64,3)(inputs)
    x=layers.BatchNormalization()(x)
    x=tf.keras.activations.relu(x)

    x=layers.Conv2D(64,3)(inputs)
    x=layers.BatchNormalization()(x)
    x=tf.keras.activations.relu(x)
    x=layers.Flatten()(x)
    x=layers.Dense(64,activation='relu')(x)
    outputs=layers.Dense(10)(x)

    model=keras.Model(inputs=inputs,outputs=outputs)
    return model

model=Soumen_model()
print(model.summary())

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy'],
)

model.fit(x_train,y_train,batch_size=64 ,epochs=10,verbose=2)
model.evaluate(x_test,y_test,batch_size=64,verbose=2)          