from interpret.blackbox import LimeTabular
from interpret import show
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import shap
import pandas as pd
from lime.lime_text import LimeTextExplainer
import lime.lime_tabular
import numpy as np
import random
import eli5
import pickle
import os.path
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


class rf:
    def __init__(self,x_train, x_test, y_train, y_test,x_features):
        self.x_test=x_test
        self.y_test=y_test
        self.x_train=x_train
        self.y_train=y_train
        self.x_features=x_features

    def rfTrain(x_train, x_test, y_train, y_test,x_features):

        rf = RandomForestClassifier(n_estimators=76,criterion="gini",min_samples_split=5,max_features="log2",random_state=1).fit(x_train,y_train)

        predictions = rf.predict(x_test)
        print(len(predictions))
        print(predictions)

        # predictions=[]
        # for i in range(len(x_test)):
        #     # Satırdaki özellik değerlerini bir numpy dizisine dönüştür
        #     user_features = np.array(x_test[i, 0:])
        #
        #     # Modeli kullanarak tahmin et
        #     prediction = rf.predict(user_features.reshape(1, -1))
        #     print(int(prediction[0]))
        #     # Tahmin sonucunu listeye ekle
        #     predictions.append(int(prediction[0]))
        # print(len(predictions))

        # Tahmin sonuçlarını submission dosyasına yaz
        submission_data = pd.DataFrame({"user_id": x_test["user_id"], "moved_after_2019": predictions})
        submission_data.to_csv("sub.csv", index=False)


        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(x_train),
                                                           feature_names=x_features,
                                                           verbose=True,
                                                           class_names=["Diabetes_binary"],
                                                           mode="regression")
        a=random.randint(0,len(y_test))
        ex = explainer.explain_instance(x_test[a], rf.predict, num_features=len(x_features))
        print(f"{a} Numaralı kişinin gerçek değeri: {y_test.iloc[a][0]}")
        print(ex.as_list())
        chance={
            "HighBP":"High blood pressure",
            "HighChol":"High Cholesterol",
            "CholCheck":"Cholesterol Check",
            "BMI":"Body mass index",
            "Smoker":"Smoker",
            "Stroke":"Stroke",
            "HeartDiseaseorAttack":"Heart Diseaseor Attack",
            "PhysActivity":"Physical Activity",
            "Fruits":"Fruits",
            "Veggies":"Veggies",
            "HvyAlcoholConsump":"Heavy Alcohol Consumption",
            "AnyHealthcare":"Any Healthcare",
            "NoDocbcCost":"No Doctor but could Cost",
            "GenHlth":"Gen Health",
            "MentHlth":"Mental Health",
            "PhysHlth":"Physical Health",
            "DiffWalk":"Difficult Walk",
            "Sex":"Sex",
            "Age":"Age",
            "Education":"Education",
            "Income":"Income"
        }
        weights = eli5.show_weights(rf, feature_names=x_features.tolist())
        results=pd.read_html(weights.data)[0]
        data=results.values
        i=0
        attlist=[]
        #vallist=[]

        while i<5:
            #txt=data[i][0]
            #x=txt.split()
            #vallist.insert(i,x)
            attlist.insert(i,chance[data[i][1]])
            i=i+1

        txt = f'I am meta person. I was created to make a global explanation based on the Eli five model for each example given to me. The five most important features of the example data.' \
              f'Here is what I am working on. ' \
              f'In order of priority, the most important features are: {attlist[0]}, {attlist[1]}, {attlist[2]},{attlist[3]} and {attlist[4]}.' \
              f'These features have a significant impact on the classification result.'

        return txt

        # n_estimators=[95,100,105]                 #GridSearchCV
        # criterion=["gini", "entropy", "log_loss"]
        # min_samples_split=[1,2,3]
        # max_features=["sqrt", "log2", None]
        # param_grid = dict(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split,
        #                   max_features=max_features)
        # grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
        # grid_result = grid.fit(x, y)
        # # Summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # print("End")


        # y_pred = rf.predict(x_test)                       #Predictions
        # score=accuracy_score(y_test, y_pred)
        # print(classification_report(y_test, y_pred))
        # print(score)







