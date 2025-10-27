from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os
import pandas as pd


os.makedirs('models',exist_ok=True)
mlflow.set_experiment("wine-quality_test")

df=pd.read_csv("data\wineQT.csv")

df['good']=(df['quality']>0).astype(int)
x=df.drop(columns=['good,quality'],axis=1)
y=df['good']

x_train,x_test,y_train,y_test=train_test_split(x,y,text_size=0.2,random_state=42)

estimators=50
with mlflow.start_run():
    model=RandomForestClassifier(n_estimators=estimators,random_state=42)
    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)

    accuracy=accuracy_score(y_test,y_pred)

    print(f'accuracy{accuracy}')

    mlflow.log_param('n_estimaors',estimators)
    mlflow.log_metric('accuracy',accuracy)
    mlflow.sklearn.log_model(model,artifact_path="models")

    
