import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets ,svm ,metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix



#Splitting of Dataset into Train and Test set
def Tain_Test_Split_file(file,arg):
	#from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(file.data, file.target, test_size=0.2,random_state=109) # 80% training and 20% test
	print("The X_train value :",X_train)
	print("\n")
	print("The X_test value :",X_test)
	print("\n")
	print("The y_train value :",y_train)
	print("\n")
	print("The y_test value :",y_test)
	SVM_Classifier(arg ,X_train,y_train ,X_test,y_test)	# call SVM_Classifier Function


	
#Create a svm Classifier for data analysis
def SVM_Classifier(arg,X_train,y_train,X_test,y_test):	
	svc_classifier = svm.SVC(kernel=arg) # Linear Kernel or Gausian Kernal or polynomial kernal or sigmoid kernal  provided as command line arguments
	#Train the model using the training sets
	svc_classifier.fit(X_train, y_train)
	#Predict the response for test dataset
	y_pred = svc_classifier.predict(X_test)
	print("\n")
	print("The y_pred value :",y_pred)
	print("\n")
	#Print Confusion matrix and accuraccy 
	cnf_matrix=confusion_matrix(y_test, y_pred)
	#print("Confusion matrix for"+" " + arg +" "+"kernal ",confusion_matrix(y_test, y_pred))
	print("Confusion matrix for"+" " + arg +" "+"kernal ",cnf_matrix)
	print(classification_report(y_test, y_pred))
	print("\n")
	print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
	print("Precision:",metrics.precision_score(y_test, y_pred))
	print("Recall:",metrics.recall_score(y_test, y_pred))
	#plt.scatter(X_test,y_pred)
	#plt.plot(X_test,y_pred, color='blue',linewidth=1) 
	plt.plot(X_test,y_pred, color='green', marker='*', linestyle='dashed',linewidth=1, markersize=12)
	plt.show()
	print("\n")
	# call Heat_Map function 
	Heat_Map(cnf_matrix)
	# call ROC_Curve Function	
	ROC_Curve(arg,y_pred,y_test)



#Visualizing Confusion Matrix using Heatmap ,
#Diagonal values represent accurate predictions, while non-diagonal elements are inaccurate predictions
def Heat_Map(cnf_matrix):
	class_names=[0,1] # name  of classes
	fig, ax = plt.subplots()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names)
	plt.yticks(tick_marks, class_names)
	# create heatmap
	sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
	ax.xaxis.set_label_position("top")
	plt.tight_layout()
	plt.title('Confusion matrix', y=1.1)
	plt.ylabel('Actual label')
	plt.xlabel('Predicted label')
	#Text(0.5,257.44,'Predicted label')



#Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false positive rate. 
#It shows the tradeoff between sensitivity and specificity.
def ROC_Curve(arg,y_pred,y_test):
	#y_pred_proba = logreg.predict_proba(X_test)[::,1]
	#fpr=False Positive Rate  and  tpr= True Positive Rate
	fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
	auc = metrics.roc_auc_score(y_test, y_pred)
	plt.plot(fpr,tpr,label="SVM Kernal ="+arg+"  " +"auc="+str(auc),color='red')
	plt.legend(loc=4)
	plt.show()





# Read the Dataset from sckit learn dataset library for Breast Cancer Analysis
def Read_Dataset():
	cancer = datasets.load_breast_cancer()
	print("Description about the cancer data set:", cancer)
	print("\n")
	print(" see the Features: ", cancer.feature_names)
	print("\n")
	print("see the Labels: ",cancer.target_names)
	print("\n")
	print ("File shape is :",cancer.data.shape)
	print("\n")
	#print(cancer.data)
	print("\n")
	print("Target value is",cancer.target)
	return cancer



def main():
	n = len(sys.argv)
	print("The number of command line arguments n =",n)
	file=Read_Dataset() # call the function Read_Dataset as command line arguments
	for i in range(1,n):
		kernal=sys.argv[i] 
		Tain_Test_Split_file(file,kernal)	
	
main()
