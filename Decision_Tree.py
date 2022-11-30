import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns


def Decision_Tree(file_path_train,file_path_test):
    df_train=pd.read_csv(file_path_train)
    print(df_train)
    # Seperate Feature columns and target coulmns
    X_train=df_train.drop('condition',axis='columns')
    y_train=df_train['condition']
    print("Features of Train_Data (X_train)) = \n ",X_train,"\n")
    print("Target/Class Labels of Train_Data (y_train)) = \n",y_train ,"\n")
  
    # Now Train our model
    model=tree.DecisionTreeClassifier()
    #model=model.fit(inputs,target)
    model=model.fit(X_train,y_train)
    print(model)
    # Testing
    df_test=pd.read_csv(file_path_test)
    print(df_test)
    X_test=df_test.drop('condition',axis='columns')
    print("Features of Test_Data (X_tset)) = \n ",X_test,"\n")
    y_test=df_test['condition']
    y_pred =model.predict(X_test)
    print("Target/Class Labels of Test_Data (y_test)) = \n",y_test ,"\n")
    print("Y_pred==>\n",y_pred,"\n")
    print("Accuracy: ==",metrics.accuracy_score(y_test, y_pred))
    print("Confusion Matrix: = \n",confusion_matrix(y_test, y_pred),"\n")
    print("Classification Report : = \n",classification_report(y_test, y_pred),"\n")
    plot_confusion_matrix(model, X_test, y_test)  
    plt.show()

    # Calculating the correlation matrix
    corr = df_train.corr()
    # Generating a heatmap
    sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns)
    plt.show()
    pass

def main():
    file_path_train=sys.argv[1]
    file_path_test=sys.argv[2]
    Decision_Tree(file_path_train,file_path_test)
    pass

main()