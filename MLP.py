from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import lime.lime_tabular
import numpy as np
import random
import pickle
import os.path
import pandas as pd


class mlp:
    def __init__(self,x_train, x_test, y_train, y_test):
        self.x_test=x_test
        self.y_test=y_test
        self.x_train=x_train
        self.y_train=y_train

    def mlpTrain(x_train, x_test, y_train, y_test,x_features):
        file_exists = os.path.exists('mlpTrained.sav')
        filename = 'mlpTrained.sav'

        if file_exists== True:
            mlp = pickle.load(open(filename, 'rb'))  #Load

        else:
            mlp = MLPClassifier(hidden_layer_sizes=49,learning_rate="constant",max_iter=200, solver="adam", activation="logistic",early_stopping=True, random_state=0).fit(x_train,y_train)
            pickle.dump(mlp, open(filename, 'wb'))   #Save



        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(x_train),
                                                           feature_names=x_features,
                                                           verbose=True,
                                                           class_names=["Diabetes_binary"],
                                                           mode="regression")
        a=random.randint(0,len(y_test))
        #a=159

        ex = explainer.explain_instance(x_test[a], mlp.predict, num_features=len(x_features))

        print(f"{a} Numaralı kişinin gerçek değeri: {y_test.iloc[a][0]}")
        print(ex.predicted_value) #right value
        results=ex.as_list()
        ex.save_to_file('lime_0.html') #Graphic Results
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
        i=0
        attlist = []
        res=ex.predicted_value.astype(int)

        if res == 0:
            res = "not diabetes "
        else:
            res = "diabetes"
        while i<5:
            txt=str(results[i][0])
            x=txt.split()

            if len(x)==3:
                attlist.insert(i, chance[x[0]])
            else:
                attlist.insert(i, chance[x[2]])

            i=i+1

        txt = f'I am meta person. I was created to make a local explanation based on the LIME model for each example given to me. The five most important features of the example data.' \
              f'Here is what I am working on. According to data; The person has{res}.' \
              f'In order of priority, the most important features are: {attlist[0]}, {attlist[1]}, {attlist[2]},{attlist[3]} and {attlist[4]}.' \
              f'These features have a significant impact on the classification result.'


        return txt
        # y_pred = mlp.predict(x_test)
        # print(classification_report(y_test, y_pred))

        # mlp = MLPClassifier()  # .fit(x_train, y_train)
        # print("start")
        # hidden_layer_sizes = [100, 120]
        # activation = ["identity", "logistic", "tanh", "relu"]
        # solver = ["lbfgs", "sgd", "adam"]
        # learning_rate = ["constant", "invscaling", "adaptive"]
        # max_iter = [190, 200]
        #
        # param_grid = dict(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
        #                   learning_rate=learning_rate, max_iter=max_iter)
        # grid = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, n_jobs=-1)
        # grid_result = grid.fit(x, y)
        # # Summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # print("End")