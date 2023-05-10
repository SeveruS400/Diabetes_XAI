from eli5.sklearn.explain_prediction import explain_prediction_tree_regressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import shap
from interpret import show
import lime.lime_tabular
import numpy as np
import random
import pandas as pd
from eli5 import show_weights
import eli5
from IPython.display import display

class xgb:
    def __init__(self,x_train, x_test, y_train, y_test):
        self.x_test=x_test
        self.y_test=y_test
        self.x_train=x_train
        self.y_train=y_train

    def xgbTrain(x_train, x_test, y_train, y_test):
        xgb= XGBClassifier(subsample=0.3,learning_rate=0.1,n_estimators=10,loss="log_loss",criterion="friedman_mse",max_features="auto").fit(x_train,y_train)
        y_pred = xgb.predict(x_test)                       #Predictions
        score = accuracy_score(y_test, y_pred)
        print(score)
        # a = random.randint(0, len(y_test))
        # if a==1:#Lime
        #     explainer = lime.lime_tabular.LimeTabularExplainer(np.array(x_train),
        #                                                        feature_names=x_features,
        #                                                        verbose=True,
        #                                                        class_names=["Diabetes_binary"],
        #                                                        mode="regression")
        #
        #     ex = explainer.explain_instance(x_test[a], xgb.predict, num_features=len(x_features))
        #     print(f"{a} Numaralı kişinin gerçek değeri: {y_test.iloc[a][0]}")
        #     print(ex.as_list())
        #
        # else:#ELI5
        #     print("baş")
        #     # eli=eli5.show_prediction(xgb, x_test[a],
        #     #                     feature_names=list(x_features),
        #     #
        #     #                     show_feature_values=True)
        #     # b=pd.read_html(str(display(eli.data)))
        #     # print(b)
        #     weights=eli5.explain_weights(xgb, feature_names=x.columns.tolist())
        #
        #     for i,weight in enumerate(weights):
        #         print("{}.{}.{:.2f}".format(i+1,weight.feature,weight.weight))
        #
        #     print(f"{a} Numaralı kişinin gerçek değeri: {y_test.iloc[a][0]}")

        # y_pred= xgb.predict(x_test)
        # print(classification_report(y_test, y_pred))

        # subsample=[0.3,0.4]
        # learning_rate=[0.1]
        # n_estimators=[100]
        # loss=["log_loss"]
        # criterion=["friedman_mse"]
        # max_features=["auto"]
        #
        # param_grid = dict(max_features=max_features,criterion=criterion,loss=loss,n_estimators=n_estimators,subsample=subsample,learning_rate=learning_rate)
        #
        # grid = GridSearchCV(estimator=xgb, param_grid=param_grid, cv = 3, n_jobs=-1)
        #
        # start_time = time.time()
        # grid_result = grid.fit(x, y)
        #
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # print("Execution time: " + str((time.time() - start_time)) + ' ms')
        # Best: 0.866517 using {'criterion': 'friedman_mse', 'learning_rate': 0.1, 'loss': 'log_loss', 'max_features': 'auto', 'n_estimators': 100, 'subsample': 0.3}