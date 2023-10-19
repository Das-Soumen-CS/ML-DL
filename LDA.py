# necessary import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn import metrics
import seaborn as sns
 
# read dataset from URL
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cls = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
#dataset = pd.read_csv(url, names=cls)
dataset=pd.read_csv("C:\\Users\\DELL\\Desktop\\Iris data.csv",names=cls)
print(dataset)

# divide the dataset into class and target variable
#X = dataset.iloc[:, 0:4].values # Run Loop froom 0 to 4-1=3
X=dataset.drop('Class', axis=1)
print("Features = \n",X ,"\n")
# Remove column name 'Class'
y=dataset["Class"]
#y = dataset.iloc[:, 4].values #pick the Last column
print("Class label = \n",y ,"\n")

# Preprocess the dataset and divide into train and test
sc = StandardScaler()
X = sc.fit_transform(X)
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
# Training data set
print("X_train= \n",X_train)
print("X_test= \n",X_test)
#Test data set
print("y_train=\n",y_train)
print("y_test=\n",y_test)

# apply Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)
 
# plot the scatterplot
plt.scatter(
    X_train[:,0],X_train[:,1],c=y_train,cmap='rainbow',
  alpha=0.7,edgecolors='b'
)
plt.show()
 
# classify using random forest classifier
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
 
# print the accuracy and confusion matrix
print('Accuracy : ' + str(accuracy_score(y_test, y_pred)))
conf_m = confusion_matrix(y_test, y_pred,labels=classifier.classes_)
print(conf_m)
# Classification Report
print(classification_report(y_test, y_pred))
#Plot the confusion matrix.
sns.heatmap(conf_m, 
            annot=True,
            fmt='g', 
            xticklabels=['Iris-setosa','Iris-versicolor','Iris-virginica'],
            yticklabels=['Iris-setosa','Iris-versicolor','Iris-virginica'])
plt.ylabel('Prediction',fontsize=12)
plt.xlabel('Actual',fontsize=12)
plt.title('Confusion Matrix',fontsize=17)
plt.show()