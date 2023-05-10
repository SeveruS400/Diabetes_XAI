from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import lime.lime_tabular
import numpy as np
import random

class knn:
    def __init__(self,x_train, x_test, y_train, y_test):
        self.x_test=x_test
        self.y_test=y_test
        self.x_train=x_train
        self.y_train=y_train



    def knnTrain(x_train, x_test, y_train, y_test,x_features):
        knn = KNeighborsClassifier(n_neighbors=50,algorithm="auto",weights="uniform").fit(x_train,y_train)
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(x_train),
                                                           feature_names=x_features,
                                                           verbose=True,
                                                           class_names=["Diabetes_binary"],
                                                           mode="regression")
        a=random.randint(0,len(y_test))
        ex = explainer.explain_instance(x_test[a], knn.predict, num_features=len(x_features))
        print(f"{a} Numaralı kişinin gerçek değeri: {y_test.iloc[a][0]}")
        print(ex.as_list())

        y_pred = knn.predict(x_test)
        print(classification_report(y_test, y_pred))

    # dual=["uniform","distance"]
    # algorithm=["auto", "ball_tree", "kd_tree", "brute"]
    # max_iter=[5,10,30,50,100,200]
    #
    # param_grid = dict(dual=dual,algorithm=algorithm,max_iter=max_iter)
    #
    # grid = GridSearchCV(estimator=knn, param_grid=param_grid, cv = 3, n_jobs=-1)
    #
    # grid_result = grid.fit(x, y)
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # Best: 0.862405 using {'algorithm': 'auto', 'n_neighbors': 50, 'weights': 'uniform'}