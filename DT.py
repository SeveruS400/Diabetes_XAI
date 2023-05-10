from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import lime.lime_tabular
import numpy as np
import random

class dt:
    def __init__(self,x_train, x_test, y_train, y_test):
        self.x_test=x_test
        self.y_test=y_test
        self.x_train=x_train
        self.y_train=y_train

    def dtTrain(x_train, x_test, y_train, y_test,x_features):
        dt=DecisionTreeClassifier(random_state=0,criterion="entropy",splitter="best",max_features="auto").fit(x_train,y_train)
        explainer = lime.lime_tabular.LimeTabularExplainer(np.array(x_train),
                                                           feature_names=x_features,
                                                           verbose=True,
                                                           class_names=["Diabetes_binary"],
                                                           mode="regression")
        a=random.randint(0,len(y_test))
        ex = explainer.explain_instance(x_test[a], dt.predict, num_features=len(x_features))
        print(f"{a} Numaralı kişinin gerçek değeri: {y_test.iloc[a][0]}")
        print(ex.as_list())
        y_pred = dt.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        print(score)
        print(classification_report(y_test, y_pred))

        # criterion=["gini", "entropy", "log_loss"]
        # splitter=["best","random"]
        # max_features=["auto", "sqrt", "log2"]
        #
        # param_grid = dict(splitter=splitter,max_features=max_features,criterion=criterion)
        #
        # grid = GridSearchCV(estimator=dt, param_grid=param_grid, cv = 3, n_jobs=-1)
        #
        # start_time = time.time()
        # grid_result = grid.fit(x, y)
        # # Summarize results
        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # print("Execution time: " + str((time.time() - start_time)) + ' ms')
        # Best: 0.805842 using {'criterion': 'entropy', 'max_features': 'auto', 'splitter': 'best'}