import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
#load wine dataset from sk
from sklearn.datasets import load_wine
import dagshub


df=load_wine()
x=df.data
y=df.target
X_train, X_test, y_train, y_test = train_test_split(  x, y, test_size=0.30, random_state=42)

n_estimator=67
max_depth=5
dagshub.init(repo_owner='dragi-8',repo_name='mlflow-tutorial',
             mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/dragi-8/mlflow-tutorial.mlflow")
mlflow.set_experiment("experiment-2")
with mlflow.start_run():
    rf=RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth)

    rf.fit(X_train,y_train)
    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('n_estimator',n_estimator)
    mlflow.log_param('max_depth',max_depth)

    print(accuracy)
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True,xticklabels=df.target_names, yticklabels=df.target_names, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    mlflow.log_artifact('confusion_matrix.png')

    mlflow.log_artifacts(__file__)

    mlflow.set_tags({'author':'abdullah','model':'RandomForestClassifier'})
    mlflow.sklearn.log_model(rf,'model')
