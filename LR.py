from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import shap
import pandas as pd
from interpret import show
import lime.lime_tabular
import numpy as np
import random

class lr:
    def __init__(self,x_train, x_test, y_train, y_test):
        self.x_test=x_test
        self.y_test=y_test
        self.x_train=x_train
        self.y_train=y_train

    def lrTrain(x_train, x_test, y_train, y_test,x_features):
        #x_train_summary = shap.kmeans(x_train, 2)
        lr = LogisticRegression(C=1.5,max_iter=100,penalty="l2",solver="saga").fit(x_train,y_train)
        y_pred = lr.predict(x_test)                       #Predictions
        score = accuracy_score(y_test, y_pred)
        print(score)


        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(x_train),
                                                           feature_names=x_features,
                                                           verbose=True,
                                                           class_names=["Diabetes_binary"],
                                                           mode="regression")
        a=random.randint(0,len(y_test))
        ex = explainer.explain_instance(x_test[a], lr.predict, num_features=len(x_features))
        print(f"{a} Numaralı kişinin gerçek değeri: {y_test.iloc[a][0]}")
        print(ex.as_list())

        y_pred = lr.predict(x_test)
        print(classification_report(y_test, y_pred))


        # ex = shap.KernelExplainer(lr.predict, x_train_summary)
        # shap_values = ex.shap_values(x)
        # shap.force_plot(ex.expected_value[1], shap_values[1][0,:], x.iloc[0, :])


        # penalty=["l1", "l2", "elasticnet", "none"]
        # solver=["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
        # max_iter=[100,110]
        # C = [1.0,1.5,2.0]
        # param_grid = dict(max_iter=max_iter,C=C,penalty=penalty)
        #
        # grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 3, n_jobs=-1)
        #
        # start_time = time.time()
        # grid_result = grid.fit(x, y)
        # # Summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # print("Execution time: " + str((time.time() - start_time)) + ' ms')
        # Best: 0.862949 using {'C': 1.5, 'dual': False, 'max_iter': 100}
        # explainer = shap.KernelExplainer(lr.predict_proba, x_train)
        # shap_values = explainer.shap_values(x_test)
        # show(shap.force_plot(explainer.expected_value[0], shap_values[0], x_test))