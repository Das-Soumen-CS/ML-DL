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

# Sequential API
model= tf.keras.Sequential(
    [
        keras.Input(shape=(32,32,3)),    # heihht =32 ,Width=32 , No of Channel =3 (RGB)
        layers.Conv2D(32,3,padding='valid' ,activation='relu',name='First_Layer'),       # No of output Channels = 32  , Kernal Size=3 , Padding = valid (default)or same
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64,3,padding='same',activation='relu',name='Second_Layer'),
        layers.MaxPooling2D(pool_size=(3,3)),
        layers.Conv2D(128,3,padding='valid',activation='relu',name='Third_Layer'),
        layers.Flatten(name='Fourth_Layer'),
        layers.Dense(64,activation='softmax',name='Fifth_Layer'),
        layers.Dense(10,name='output_Layer'),
    ]
)

print(model.summary())

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy'],
)

model.fit(x_train,y_train,batch_size=64 ,epochs=10,verbose=2)
model.evaluate(x_test,y_test,batch_size=64,verbose=2)
       