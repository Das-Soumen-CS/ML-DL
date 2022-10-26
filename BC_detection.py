
import numpy as np
import sys
from numpy.core.fromnumeric import size
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

df=pd.read_csv(sys.argv[1])
print("\n#Description about dataset::\n\n",df.head(8))

print("\n#Dimension of dataset ::",df.shape)

print("\n#Count the empty (NaN, NAN, na) values in each column::\n")
print(df.isna().sum())

#print("\nDrop the column with all missing values (na, NAN, NaN)::\n")
df=df.dropna(axis=1)
#print(df.dropna(axis=1))
print("\n#New Dimension of dataset after drop the column with all missing values (na, NAN, NaN) is:: ",df.shape)

print("\n#To find cardinality of Benign and Malignant cells::\n ")
print(df['diagnosis'].value_counts())

sns.countplot(x ='diagnosis',hue = "diagnosis", data = df)
plt.show()

print("\nFeatures datatype description:::\n")
print(df.dtypes)


from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)
print("\nEncode the categorical data from column ‘diagnosis’ as M=>1 and B=>0\n")
print(labelencoder_Y.fit_transform(df.iloc[:,1].values))


#sns.pairplot(df, hue="diagnosis")
#plt.show()


df.corr()
plt.figure(figsize=(20,20))  
sns.heatmap(df.corr(),linewidth=0.3,vmin= 0.0, vmax=1.0,annot = True,annot_kws={'size':6.5},fmt='.0%')
plt.show()

X = df.iloc[:, 2:31].values 
Y = df.iloc[:, 1].values 
print("Indepedent set",X)
print("Dependent set",Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =float(sys.argv[2]), random_state =0)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print(X_train)
X_test = sc.transform(X_test)
print(X_test)


def models(X_train,Y_train):
  
  #Using Logistic Regression 
  from sklearn.linear_model import LogisticRegression
  log = LogisticRegression(random_state = 0)
  log.fit(X_train, Y_train)
  
  #Using KNeighborsClassifier 
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
  knn.fit(X_train, Y_train)

  #Using SVC linear
  from sklearn.svm import SVC
  svc_lin = SVC(kernel = 'linear', random_state = 109)
  svc_lin.fit(X_train, Y_train)
  y_pred = svc_lin.predict(X_test)
  cnf_matrix=confusion_matrix(Y_test, y_pred)
  print(classification_report(Y_test, y_pred))
  plt.plot(X_test,y_pred, color='green', marker='*', linestyle='dashed',linewidth=1, markersize=12)
  plt.show()
  print("\n")
  class_names=[0,1] # name  of classes
  fig, ax = plt.subplots()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names)
  plt.yticks(tick_marks, class_names)
  sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
  ax.xaxis.set_label_position("top")
  plt.tight_layout()
  plt.title('Confusion matrix', y=1.1)
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  plt.show()
  
  #Using SVC poly
  from sklearn.svm import SVC
  svc_poly = SVC(kernel = 'poly', random_state = 0)
  svc_poly.fit(X_train, Y_train)
  
  #Using SVC sigmoid
  from sklearn.svm import SVC
  svc_sigmoid = SVC(kernel = 'sigmoid', random_state = 0)
  svc_sigmoid.fit(X_train, Y_train)
  

  #Using SVC rbf
  from sklearn.svm import SVC
  svc_rbf = SVC(kernel = 'rbf', random_state = 0)
  svc_rbf.fit(X_train, Y_train)

  #Using GaussianNB 
  from sklearn.naive_bayes import GaussianNB
  gauss = GaussianNB()
  gauss.fit(X_train, Y_train)

  #Using DecisionTreeClassifier 
  from sklearn.tree import DecisionTreeClassifier
  tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
  tree.fit(X_train, Y_train)

  #Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm
  from sklearn.ensemble import RandomForestClassifier
  forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
  forest.fit(X_train, Y_train)
  
  #print model accuracy on the training data.
  print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
  print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
  print('[2]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
  print('[3]Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
  print('[4]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
  print('[5]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
  print('[6]Support Vector Machine (Ploynomial Classifier) Training Accuracy:', svc_poly.score(X_train, Y_train))
  print('[7]Support Vector Machine (Sigmoid Classifier) Training Accuracy:', svc_sigmoid.score(X_train, Y_train))
  print('[8]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
  
  return log, knn, svc_lin, svc_poly,svc_sigmoid,svc_rbf, gauss, tree, forest

model = models(X_train,Y_train)


from sklearn.metrics import confusion_matrix
for i in range(len(model)):
  cm = confusion_matrix(Y_test, model[i].predict(X_test))
  
  TN = cm[0][0]
  TP = cm[1][1]
  FN = cm[1][0]
  FP = cm[0][1]
  print('\nConfusion matrix for Model=>',i)
  print(cm)
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
for i in range(len(model)):
  print('Model ',i)
  #Check precision, recall, f1-score
  print( classification_report(Y_test, model[i].predict(X_test)) )
  #Another way to get the models accuracy on the test data
  print("Test_accuracy =", accuracy_score(Y_test, model[i].predict(X_test)))
  print()#Print a new line

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
    
