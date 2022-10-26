import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices =tf.config.list_physical_devices('GPU')
#m =tf.config.experimental.set_memory_growth(physical_devices[0],True)
print("No of GPU  =",len(physical_devices))

(x_train,y_train) ,(x_test,y_test)=mnist.load_data()  # Load Data

x_train=x_train.reshape(-1,28,28,1).astype("float32")/255.0   #Normalize data
x_test=x_test.reshape(-1,28,28,1).astype("float32")/255.0

#model definition
model=tf.keras.Sequential(
    [
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(63,3,padding='same'),
        layers.ReLU(),
        layers.Conv2D(128,3,padding='same'),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dense(10),
    ],
    name='model',
    )

class CustomFit(keras.Model):
    def __init__(self,model):
        super(CustomFit,self).__init__()
        self.model=model

    def compile(self,optimizer,loss):   #Custom compile 
        super(CustomFit,self).compile()
        self.optimizer=optimizer
        self.loss=loss  


    def train_step(self,data):  #Custom Train
        x,y=data

        with tf.GradientTape() as tape:   #Gradient Descent
            y_pred=self.model(x,training=True)
            #loss=self.compiled_loss(y,y_pred)
            loss=self.loss(y,y_pred) # For custom loss 
           

        training_vars=self.trainable_variables
        gradients=tape.gradient(loss,training_vars)

        self.optimizer.apply_gradients(zip(gradients,training_vars))
        #self.compiled_metrics.update_state(y,y_pred)
        #return{m.name:m.result() for m in self.metrics}
        acc_metric.update_state(y,y_pred)  # For custom 
        return{"loss":loss,"accuracy":acc_metric.result()} #For Custom loss explicitly


    def test_step(self,data):
        x,y=data
        y_pred=self.model(x,training=False)
        loss=self.loss(y,y_pred)
        acc_metric.update_state(y,y_pred)
        return{"loss":loss,"accuracy":acc_metric.result()}


acc_metric=keras.metrics.SparseCategoricalCrossentropy(name='accuracy')

training=CustomFit(model)

training.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #metrics=["accuracy"], # for Custom Loss remove metrices
)

training.fit(x_train,y_train,batch_size=32,epochs=100)  # fit works on Train set

training.evaluate(x_test,y_test,batch_size=32)  # evaluate works on Test Set ,hence need to define a Test function
