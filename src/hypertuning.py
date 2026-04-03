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
from sklearn.datasets import load_breast_cancer
import dagshub#import grid search cv
from sklearn.model_selection import GridSearchCV


data=load_breast_cancer()
x=pd.DataFrame(data.data, columns=data.feature_names)
y=pd.Series(data.target,name='target')

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, random_state=42)

param_grid={
    'n_estimators':[0,50,100,150],
    'max_depth':[None,5,10,15,20,30]}

rf=RandomForestClassifier(random_state=42)
search_model=GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)
# search_model.fit(train_x,train_y)

# best_params=search_model.best_params_
# print(best_params)
# best_score=search_model.best_score_
# print(best_score)


dagshub.init(repo_owner='dragi-8',repo_name='mlflow-tutorial',
             mlflow=True)
mlflow.set_experiment("Random Forest Hyperparameter Tuning")
with mlflow.start_run() as parent:
    search_model.fit(train_x,train_y)
    best_params=search_model.best_params_
    best_score=search_model.best_score_
    for i in range(len(search_model.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(search_model.cv_results_['params'][i])
            mlflow.log_metric("mean_test_score", search_model.cv_results_['mean_test_score'][i])
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_score)

    train_df=train_x.copy()
    train_df['target']=train_y
    train_df=mlflow.data.from_pandas(train_df)

    test_df=test_x.copy()
    test_df['target']=test_y
    test_df=mlflow.data.from_pandas(test_df)

    mlflow.log_input(train_df, "training")
    mlflow.log_input(test_df, "testing")

    mlflow.sklearn.log_model(search_model.best_estimator_,'best_rf_model')

    mlflow.set_experiment_tags({"model":"Random Forest","dataset":"Breast Cancer Wisconsin","method":"Grid Search CV"})

    print(f"Best Parameters: {best_params}")
    print(f"Best Cross-Validation Accuracy: {best_score}")

