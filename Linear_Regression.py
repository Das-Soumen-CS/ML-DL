import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sys
from sklearn import datasets, linear_model 
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

   
# Load CSV and columns 
def Read_ReshapeFile(filePath):
	#To read the file path as : filePath=sys.argv[1] 
	df = pd.read_csv(filePath)
	print(df)
	Y = df['price']
	print (Y)
	X = df['lotsize']
	print (X)
	Y=Y.values.reshape(len(Y),1)
	X=X.values.reshape(len(X),1)
	# Split the data into training/testing sets
	X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0) 
	PlotFig(X_train, X_test,Y_train, Y_test)
   
# Plot outputs 
def PlotFig(X_train, X_test,Y_train, Y_test):
	plt.scatter(X_test, Y_test, color='blue') 
	plt.title('Test Data') 
	plt.xlabel('Size of House') 
	plt.ylabel('Price of House') 
	plt.xticks(()) 
	plt.yticks(()) 
	Linear_Regression(X_train, X_test,Y_train,Y_test)
	Logistic_Regression(X_train, X_test,Y_train, Y_test)
	

def Linear_Regression(X_train, X_test,Y_train,Y_test):
	Linear_regr = linear_model.LinearRegression() 
	# Train the model using the training sets 
	Linear_regr.fit(X_train, Y_train.reshape(len(Y_train),1)) 
	Y_pred=Linear_regr.predict(X_test)
	Y_pred=Y_pred.reshape(len(Y_pred),1)
	# Plot outputs 
	plt.plot(X_test,Y_pred, color='red',linewidth=3) 
	plt.show()
	#Print Confusion matrix and accuraccy 
	#print("Confusion matrix for"+" " + "Linear_Regression "+" ",confusion_matrix(Y_test, Y_pred))
	#ValueError: Classification metrics can't handle a mix of multiclass and continuous target, {So Replace Y_Pred as Y_pred.round()}
	print("Confusion matrix for"+" " + "Linear_Regression "+" ",confusion_matrix(Y_test, Y_pred.round(),normalize=None))
	print("\n")
	print("Classification Report for Linaer Regression:")
	print("\n")
	print(classification_report(Y_test, Y_pred.round(),zero_division=1))
	print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred.round()))
	print("\n")


def Logistic_Regression(X_train, X_test,Y_train, Y_test):
	logreg = linear_model.LogisticRegression()
	# fit the model with data
	logreg.fit(X_train,Y_train)
	Y_pred=logreg.predict(X_test)
	plt.plot(X_test,Y_pred, color='red',linewidth=3) 
	plt.show()
	print("\n")
	print("The y_pred value :",Y_pred)
	print("\n")
	#Print Confusion matrix and accuraccy 
	print("Confusion matrix for"+" " + "Logistic_Regression "+" ",confusion_matrix(Y_test, Y_pred))
	print("\n")
	print("Classification report for Logistic Regression:")
	print("\n")
	print(classification_report(Y_test, Y_pred,zero_division=1))
	print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred.round()))
	print("\n")

def main():
	n = len(sys.argv)
	print("The number of command line arguments n =",n)
	filePath=sys.argv[1]
	Read_ReshapeFile(filePath)
	
	

main()