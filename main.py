import itertools
import time

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy import stats
from sklearn.preprocessing import scale, MinMaxScaler

# trainingPath = "data/TrainOnMe_RE.csv"
# evaluationPath = "data/EvaluateOnMe_RE.csv"

# Most testing has been done on this .
# trainingPath = "data/TrainOnMe-2.csv"
# evaluationPath = "data/EvaluateOnMe-2.csv"

trainingPath = "data/train3.csv"
evaluationPath = "data/eval3.csv"

trainingPath = "data/TrainOnMe-4.csv"
evaluationPath = "data/EvaluateOnMe-4.csv"

config = {"nComponents": 22}


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


def is_float(element: str) -> bool:
    try:
        float(element)
        return True
    except ValueError:
        return False


def readData(filepath=""):
    return pd.read_csv(filepath, encoding='utf-8', index_col=0)


def removeNoise(trainingData, evaluationData):
    # nCols = len(evaluationData.columns)
    # I'm going to use the evaluation data to filter my training data. I'm assuming that the evaluationData is noise free.
    # Return DataFrame with duplicate rows removed.
    trainingData = trainingData.drop_duplicates()

    # Remove rows that simply is broken, i.e. some row that are missing values.
    trainingData = trainingData.dropna()

    # Don't think we need the first column with the index. TODO THIS COULD CHANGE

    for cName in evaluationData.columns:

        uniqueValues = evaluationData[cName].unique()

        if len(uniqueValues) == 1:
            del trainingData[cName]
            del evaluationData[cName]
        else:
            if evaluationData[cName].dtype.name == 'object':

                trainingData = trainingData[trainingData[cName].isin(uniqueValues)]
                trainingData[cName].replace(uniqueValues, list(range(len(uniqueValues))), inplace=True)
                evaluationData[cName].replace(uniqueValues, list(range(len(uniqueValues))), inplace=True)

            elif evaluationData[cName].dtype.name == 'bool':
                # trainingData = trainingData[trainingData[cName].isin(["False", 'True', 'TRUE', 'FALSE'])]
                trainingData[cName].astype(bool)

                trainingData[cName] = trainingData[cName].astype(int)
                evaluationData[cName] = evaluationData[cName].astype(int)

    trainingData = trainingData[(np.abs(stats.zscore(trainingData.drop('y', axis=1))) < 4).all(axis=1)]
    # TODO CHECK 3 or 4

    # Here we do correlation values. We should plot it first in order go get som insights.  to data

    corrMatrix = trainingData.corr().abs()
    upper = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(bool))

    dropLimit = 0.90
    to_drop = [column for column in upper.columns if any(upper[column] >= dropLimit)]
    # Drop features
    trainingData.drop(to_drop, axis=1, inplace=True)
    evaluationData.drop(to_drop, axis=1, inplace=True)
    trainingData = trainingData.groupby('y').apply(lambda x: x.sample(frac=0.25))  # 0.015  # 0.15 took 100 minutes
    X_train = trainingData.drop('y', 1)
    Y_train = trainingData['y']
    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    X_train = scaling.transform(X_train)
    evaluationData = scaling.transform(evaluationData)

    pca = PCA(n_components=18).fit(X_train)
    # # These will become Numpy arrays
    X_train = pca.transform(X_train)
    evaluationData = pca.transform(evaluationData)

    return X_train, Y_train, evaluationData


if __name__ == '__main__':

    X_train, Y_train, X_eval = removeNoise(readData(trainingPath), readData(evaluationPath))
    print("Number of points to train on is: {}".format(len(X_train)))
    classifiers = [
        KNeighborsClassifier(15),
        SVC(kernel="linear", C=0.025),
        SVC(kernel="rbf", C=1.503060, gamma='auto'),
        GradientBoostingClassifier(),  # SLOW
        DecisionTreeClassifier(),
        RandomForestClassifier(criterion='entropy', n_estimators=250, max_features='sqrt'),
        MLPClassifier(alpha=0.1778439837548613, max_iter=1000),
        AdaBoostClassifier(),
        AdaBoostClassifier(GaussianNB()),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        LinearDiscriminantAnalysis()
    ]


    def gridSearch():

        print("Staring grid search")

        def svc():
            startTime = time.time()
            sC = 0
            dC = 5
            sG = 0.00001
            dG = 2.5
            C_range = np.linspace(1.593541468827725 - 0.15, 1.593541468827725 + 0.15, 500)
            gamma_range = np.linspace(sG, sG + dG, 100)
            param_grid = dict(gamma=['auto'], C=[1.503060], kernel=['linear', 'poly', 'rbf', 'sigmoid'])
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=14, verbose=10)
            grid.fit(X_train, Y_train)

            print(
                "The best parameters are %s with a score of %0.5f"
                % (grid.best_params_, grid.best_score_)
            )

            print("Tiden var {}".format(time.time() - startTime))

        def mlp():
            startTime = time.time()

            a_range = np.linspace(0, 0.19684194472866112 + 1, 250)
            param_grid = dict(alpha=a_range)
            cv = StratifiedShuffleSplit(n_splits=6, test_size=0.2, random_state=42)
            grid = GridSearchCV(MLPClassifier(), param_grid=param_grid, cv=cv, n_jobs=14, verbose=10)
            grid.fit(X_train, Y_train)

            print(
                "The best parameters are %s with a score of %0.5f"
                % (grid.best_params_, grid.best_score_)
            )

            print("Tiden var {}".format(time.time() - startTime))

        def stacking():
            forStacking = [
                ('knn', KNeighborsClassifier(20)),
                ('Lsvc', SVC(kernel="linear", C=1)),
                ('rbf', SVC(kernel="rbf", C=1.503060, gamma='auto')),
                ('gbc', GradientBoostingClassifier()),
                ('dtt', DecisionTreeClassifier()),
                ('rfc', RandomForestClassifier(criterion='entropy', n_estimators=250, max_features='sqrt')),
                ('mlp', MLPClassifier(alpha=0.1778439837548613, max_iter=1000)),
                ('ada', AdaBoostClassifier()),
                ('naiveGauss', GaussianNB()),
                ('qda', QuadraticDiscriminantAnalysis(),),
                ('lda', LinearDiscriminantAnalysis())
            ]
            combos = []
            finalRes = (-10, None, None)
            for i in range(3, 5):
                t = itertools.combinations(forStacking, i)
                for c in t:
                    combos.append(list(c))

            for c in tqdm.tqdm(combos):
                clf = StackingClassifier(estimators=c, final_estimator=MLPClassifier(max_iter=1000), cv=7)
                result = cross_val_score(clf, X_train, Y_train, cv=16, n_jobs=16)
                finalRes = max((np.mean(result), list(c)), finalRes, key=lambda x: x[0])

            print(finalRes)
            cache = open("finalRes.txt", mode="w")
            cache.write(str(finalRes))
            cache.close()

        # svc()
        # mlp()
        # stacking()


    def testFunction():

        n = 14
        finalRes = (-10, "none")
        for clf in classifiers:
            result = cross_val_score(clf, X_train, Y_train, cv=n, n_jobs=14)  # verbose=5,
            print("nCV: {}\t\tMean score: {}\tStd: {}\t\tClassifier: {}\t\t\t".format(n, round(np.mean(result), 4),
                                                                                      round(np.std(result), 4),
                                                                                      clf.__class__.__name__, ))
            finalRes = max((np.mean(result), clf.__class__.__name__,), finalRes, key=lambda x: x[0])

        # estimators = [('knn', KNeighborsClassifier(n_neighbors=20)), ('Lsvc', SVC(C=1, kernel='linear')), ('mlp', MLPClassifier(alpha=0.1778439837548613, max_iter=1000)), ('naiveGauss', GaussianNB())]
        estimators = [('knn', KNeighborsClassifier(n_neighbors=20)), ('Lsvc', SVC(C=1, kernel='linear')),
                      ('mlp', MLPClassifier(alpha=0.1778439837548613, max_iter=1000)), ('ada', AdaBoostClassifier())]

        clf = StackingClassifier(estimators=estimators, final_estimator=MLPClassifier(max_iter=1250), cv=7)

        result = cross_val_score(clf, X_train, Y_train, cv=n, n_jobs=14, verbose=10)
        print("nCV: {}\t\tMean score: {}\tStd: {}\t\tClassifier: {}\t\t\t".format(n, round(np.mean(result), 4),
                                                                                  round(np.std(result), 4),
                                                                                  clf.__class__.__name__, ))

        finalRes = max((np.mean(result), clf.__class__.__name__,), finalRes, key=lambda x: x[0])
        print(finalRes)
        print("Evaluation result")

        clf.fit(X_train, Y_train)

        evalRes = clf.predict(X_eval, )
        print(*evalRes)

        cache = open("evalRes.txt", mode="w")

        for label in evalRes:
            cache.write(label + "\n")
        cache.close()

        return finalRes


    # gridSearch()

    # finalTest()

    # print("Done!")


    def finalSubmission():

        estimators = [('knn', KNeighborsClassifier(n_neighbors=20)), ('Lsvc', SVC(C=1, kernel='linear')),
                      ('mlp', MLPClassifier(alpha=0.1778439837548613, max_iter=1000)), ('ada', AdaBoostClassifier())]

        clf = StackingClassifier(estimators=estimators,
                                 final_estimator=MLPClassifier(max_iter=1250),
                                 cv=14,
                                 n_jobs=14,
                                 verbose=10)

        clf.fit(X_train, Y_train)

        evalRes = clf.predict(X_eval)

        cache = open("evalRes3.txt", mode="w")

        for label in evalRes:
            cache.write(label + "\n")
            print(label)
        cache.close()

    finalSubmission()

