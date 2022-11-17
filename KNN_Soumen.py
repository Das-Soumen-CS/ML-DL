import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def Read_dataset(file_path,neighbors):
	col_names= ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
	dataset=pd.read_csv(file_path,names=col_names)
	print("Description about the  data set:\n\n", dataset)
	X = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, 4].values
	KNN_Supervised_ML(X,y,neighbors)
	return dataset

def KNN_Supervised_ML(Attribute,Class_label,neighbors):
	#To avoid over-fitting,dividing the dataset into training and test splits
	X_train, X_test, y_train, y_test = train_test_split(Attribute,Class_label, test_size=0.20,random_state=109) 
	print("The X_train value :\n \n",X_train)
	print("\n")
	print("The X_test value :\n \n",X_test)
	print("\n")
	print("The y_train value :\n\n",y_train)
	print("\n")
	print("The y_test value :\n\n",y_test) 
	Scale_Dataset(X_train,X_test)
	Tarin_and_Predict(X_train,y_train,X_test,y_test,neighbors)

	
	#Before making any actual predictions, keep a good practice to scale the features so that all of them can be uniformly evaluated
def Scale_Dataset(X_train,X_test):
	scaler = StandardScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	print("The X_train value after scaling  :\n \n",X_train)
	print("\n")
	print("The X_test value  after scaling:\n \n",X_test)
	print("\n")

def Tarin_and_Predict(X_train,y_train,X_test,y_test,NN):
	classifier = KNeighborsClassifier(n_neighbors=NN)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	print("Prediction on Test Data: \n\n",y_pred)
	cnf_matrix=confusion_matrix(y_test, y_pred)
	clf_report=classification_report(y_test, y_pred)
	print("The confution matrix for :\n\n",cnf_matrix)
	print("The Classification report is:\n\n",clf_report)
	print("\n")
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
	print("Precision:",metrics.precision_score(y_test, y_pred,pos_label=1, average=None))
	print("Recall:",metrics.recall_score(y_test, y_pred, average=None))
	plt.plot(X_test,y_pred, color='green', marker='*', linestyle='dashed',linewidth=1, markersize=12)
	plt.show() 
	Heat_Map(cnf_matrix)
	ErrorRate_for_diff_k_val(X_train,X_test,y_train,y_test,NN)

def Heat_Map(cnf_matrix):
	cm_df = pd.DataFrame(cnf_matrix,index = ['setosa','versicolor','virginica'], columns = ['setosa','versicolor','virginica'])
	sns.heatmap(cm_df, annot=True,cmap="YlGnBu" ,fmt='g')
	plt.ylabel('Actual label')
	plt.xlabel('Predicted label')
	plt.show()

def ErrorRate_for_diff_k_val(X_train,X_test,y_train,y_test,NN):
	scores=[]
	for i in range(1,NN):
		knn = KNeighborsClassifier(n_neighbors=i)
		knn.fit(X_train, y_train)
		pred_i = knn.predict(X_test)
		scores.append(metrics.accuracy_score(y_test,pred_i))
		plt.figure(figsize=(12, 6))
		plt.plot( scores, color='red', linestyle='dashed', marker='*',markerfacecolor='blue',markersize=10)
		plt.title('Error Rate for diffrent K Values')
		plt.xlabel('K Value')
		plt.ylabel('Mean Error')
		plt.show()
		print(scores)

def main():
	n=len(sys.argv)
	print("The no of command line arguments that has provided :: " ,n)
	file_path=sys.argv[1]
	NN=int(sys.argv[2])
	file=Read_dataset(file_path,NN)
	

main()