import pandas as pd
import numpy as np 
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler ,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import sys

data_set= load_digits()
#print(data_set,"\n")
print("\n Keys are =",data_set.keys(),"\n")

print("The dimension of  dataset is =",data_set.data.shape,"\n")
print("See the First Sample in 1D form = \n",data_set.data[0],"\n")   # It is 1D numpy array 
#print("Before reshape =",data_set.data[0].shape())

# if we want to see in 2D need to reshape
print("After reshape into 2D =\n",data_set.data[0].reshape(8,8),"\n")
plt.matshow(data_set.data[0].reshape(8,8))
plt.gray()
plt.show()
# To see the Class label or Target Variables 
classlabels =np.unique(data_set.target)
print("Target values are =",classlabels,"\n")

# create a datframe 
df=pd.DataFrame(data_set.data ,columns=data_set.feature_names)   # pixel_0_0 = oth row o pixel , pixel_0_1 = oth row first pixel and so on...
print("Dataset features= \n",df.head(),"\n")
print("DataSet Description =\n",df.describe())  # notice column pixel_0_0  is unimportant as all values are zero

# Assign all featutes in variable x and target values in y
x=df
y=data_set.target

# Before building any ML model we need to Scale our features 
print("Features before scaling =\n ",x,"\n")
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)
print("Features after scaling(using StandardScaler) = \n",x_scaled,"\n")

# scale features
scaler = MinMaxScaler()
model=scaler.fit(x)
scaled_data=model.transform(x)
#x_scaled=scaler.fit_transform(x)
print("Features after scaling(using Min-MaxScaler)= \n",scaled_data,"\n")

# Split the dataset for Train (80%) and Test (20%)

x_train,x_test ,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=30)
model_1 =LogisticRegression()
model_1.fit(x_train,y_train)
print("model_1(LogisticRegression) accurracy = ",model_1.score(x_test,y_test),"\n")


# PCA when wish to retain amount of impormation interms of percentage
print("PCA when wish to retain amount of impormation interms of percentage=>>>",sys.argv[1],"\n")
print("Actual Dimensions = ",x.shape,"\n")
pca=PCA(float(sys.argv[1]))    # provide commandline argument for retaining X % of information from the dataset
x_pca=pca.fit_transform(x)
print("Reduced Dimensions = ",x_pca.shape,"\n")
# again train our data after dimensionality reduction by using PCA
x_train_pca,x_test_pca ,y_train,y_test=train_test_split(x_pca,y,test_size=0.2,random_state=30)
model_2 =LogisticRegression(max_iter=50000)
model_2.fit(x_train_pca,y_train)
print("model_2(LogisticRegression) accurracy = ",model_2.score(x_test_pca,y_test),"\n")

# PCA when we explictly tells the no of pca components we wanr for training 
print("PCA when we explictly tells the no of pca components we wanr for training =>>\n" )
pca=PCA(n_components=3)
x_pca=pca.fit_transform(x)
print("Reduced Dimensions = ",x_pca.shape,"\n")
print("Amount of Info retaining for 3 dimensions = ",pca.explained_variance_ratio_,"\n") # how much amount of information we are capturing
x_train_pca,x_test_pca ,y_train,y_test=train_test_split(x_pca,y,test_size=0.2,random_state=30)
model_3 =LogisticRegression(max_iter=50000)
model_3.fit(x_train_pca,y_train)
print("model_3(LogisticRegression) accurracy = ",model_3.score(x_test_pca,y_test),"\n")