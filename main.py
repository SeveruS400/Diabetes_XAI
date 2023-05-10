import traceback
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV


import Knn,LR,DT,RF,XGB,MLP,rivaApi

root_url = 'https://api.replicastudios.com'
client_id = 'mahirfurkan1999@gmail.com'
secret = 'Deneme153426'

txt = 'Hello world'
audio_format = 'wav'

dataFrame = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
y = pd.DataFrame(dataFrame["Diabetes_binary"].values)
x = pd.DataFrame(dataFrame.drop("Diabetes_binary", axis=1).values)

x = pd.DataFrame(x)
x = dataFrame[
    ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
     "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
     "Sex", "Age", "Education", "Income"]]
x_features = x.columns

if __name__ == '__main__':

    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)
        from sklearn.preprocessing import MinMaxScaler as Scaler
        sc=Scaler()

        x_train= sc.fit_transform(x_train)
        x_test= sc.fit_transform(x_test)

        choise=input("İşlem seç:")
        a=input("Açıklanabilirlik yöntemi")
        txt=MLP.mlp.mlpTrain(x_train, x_test, y_train, y_test,x_features)
        txt=RF.rf.rfTrain(x_train, x_test, y_train, y_test,x_features)
        if choise=="knn":
            Knn.knn.knnTrain(x_train, x_test, y_train, y_test,x_features)
        elif choise=="lr":
            LR.lr.lrTrain(x_train, x_test, y_train, y_test,x_features)
        elif choise == "dt":
            DT.dt.dtTrain(x_train, x_test, y_train, y_test,x_features)
        elif choise == "rf":
            RF.rf.rfTrain(x_train, x_test, y_train, y_test,x_features)  #ELI5
            print(txt)
            rivaApi.rivaserver(txt)
        elif choise == "xgb":
            XGB.xgb.xgbTrain(x_train, x_test, y_train, y_test,x_features,x)
        elif choise == "mlp":
            MLP.mlp.mlpTrain(x_train, x_test, y_train, y_test,x_features)   #LIME
            print(txt)
            rivaApi.rivaserver(txt)
        else:
            print("Yanlış seçim")


    except Exception as e:
        print(f'EXCEPTION: {e}')
        traceback.print_exc()

