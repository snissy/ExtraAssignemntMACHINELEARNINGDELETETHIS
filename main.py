import time

import numpy as np
import pandas as pd
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
from sklearn.preprocessing import StandardScaler

# trainingPath = "data/TrainOnMe_RE.csv"
# evaluationPath = "data/EvaluateOnMe_RE.csv"
trainingPath = "data/TrainOnMe-2.csv"
evaluationPath = "data/EvaluateOnMe-2.csv"
# trainingPath = "data/eval3.csv"
# evaluationPath = "data/train3.csv"

config = {"outliersZValue": 3, "corrDropLimit": 0.95}


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

                # TODO I think all this is needed when doing the real assignment. Seems that I got a really dirty dataset.
                cleanEvalData = [v.replace(".", "").replace(chr(8722), chr(45)) for v in evaluationData.get(cName)]
                quantitativeColum = all(is_float(element) for element in cleanEvalData)
                # fix this.
                if quantitativeColum:
                    # we convert string to float
                    #  This change.
                    cleanTrainData = [v.replace(".", "").replace(chr(8722), chr(45)) for v in trainingData.get(cName)]
                    trainingData[cName] = pd.to_numeric(cleanTrainData, downcast="float")
                    evaluationData[cName] = pd.to_numeric(cleanEvalData, downcast="float")

                else:
                    # We remove rows with values that have categories that doesn't exist in the evaluation set.
                    trainingData = trainingData[trainingData[cName].isin(uniqueValues)]
                    trainingData[cName].replace(uniqueValues, list(range(len(uniqueValues))), inplace=True)
                    evaluationData[cName].replace(uniqueValues, list(range(len(uniqueValues))), inplace=True)

            elif evaluationData[cName].dtype.name == 'bool':
                # trainingData = trainingData[trainingData[cName].isin(["False", 'True', 'TRUE', 'FALSE'])]
                trainingData[cName].astype(bool)

                trainingData[cName] = trainingData[cName].astype(int)
                evaluationData[cName] = evaluationData[cName].astype(int)

    # evaluationData.replace(["False", 'True'], [0, 1], inplace=True)
    # trainingData.replace(["False", 'True'], [0, 1], inplace=True)
    # outliers

    trainingData = trainingData[(np.abs(stats.zscore(trainingData.drop('y', axis=1))) < 5).all(axis=1)]
    # TODO CHECK 3 or 4

    # Here we do correlation values. We should plot it first in order go get som insights.  to data

    corrMatrix = trainingData.corr().abs()
    upper = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(bool))

    dropLimit = 0.9
    to_drop = [column for column in upper.columns if any(upper[column] >= dropLimit)]
    # Drop features
    trainingData.drop(to_drop, axis=1, inplace=True)
    evaluationData.drop(to_drop, axis=1, inplace=True)

    for cName in evaluationData.columns:
        if evaluationData[cName].dtype.name != 'bool':
            dataMean = evaluationData[cName].mean()
            dataStd = evaluationData[cName].std()
            evaluationData[cName] = (evaluationData[cName] - dataMean) / dataStd
            trainingData[cName] = (trainingData[cName] - dataMean) / dataStd

    X_train = trainingData.drop('y', 1)
    Y_train = trainingData['y']

    pca = PCA()
    pca.fit(evaluationData)
    newNumber = len(evaluationData.columns) - sum(
        1 for cV in pca.explained_variance_ratio_ if cV <= (1 / float(len(evaluationData.columns))) * 0.65)
    print((1 / float(len(evaluationData.columns))) * 0.65)
    # 0.05 was really good

    pca = PCA(n_components=newNumber)
    # # These will become Numpy arrays
    X_train = pca.fit_transform(X_train)
    evaluationData = pca.fit_transform(evaluationData)

    # hm = sns.heatmap(corrMatrix, annot=True)
    # hm.set(title="Correlation matrix of IRIS data\n")
    # plt.show()
    # Mean normalize data.
    # mean normalization

    # pca = PCA(n_components=X.shape[1])
    # pca.fit(X)
    # print(pca.explained_variance_ratio_)
    # TODO use sklearn SCALE FUNCTION, StandardScaler

    # for cName in evaluationData.columns:
    #     if evaluationData[cName].dtype.name != 'bool':
    #         dataMax = evaluationData[cName].max()
    #         dataMin = evaluationData[cName].min()
    #         evaluationData[cName] = (evaluationData[cName] - dataMin) / (dataMax - dataMin)
    #         trainingData[cName] = (trainingData[cName] - dataMin) / (dataMax - dataMin)

    return X_train, Y_train, evaluationData


def splitData(data, splitvalue=0.75):
    """
    :param data:
    :param splitvalue:
    :return: training set and validation set.
    """
    return train_test_split(data, test_size=1 - splitvalue)


if __name__ == '__main__':

    X_train, Y_train, X_eval = removeNoise(readData(trainingPath), readData(evaluationPath))

    # le = preprocessing.LabelEncoder()
    # le.fit(trainTargets)
    # trainTargets = le.transform(trainTargets)

    # Now we can start classify the data

    print("Number of points to train on is: {}".format(len(X_train)))
    # The best parameters are {'C': 4.259662151757895, 'gamma': 0.0383646083799769} with a score of 0.62060
    classifiers = [
        KNeighborsClassifier(15),
        SVC(kernel="linear", C=0.025),
        SVC(kernel="rbf", C=4.259662151757895, gamma=0.0383646083799769),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        GradientBoostingClassifier(),  # SLOW
        DecisionTreeClassifier(),
        RandomForestClassifier(criterion='entropy', n_estimators=250, max_features='sqrt'),
        MLPClassifier(alpha=1.4446930579460697, max_iter=1000),
        AdaBoostClassifier(),
        AdaBoostClassifier(GaussianNB()),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        LinearDiscriminantAnalysis()
    ]


    # trainData, validationData = splitData(trainData)

    # sparse coordinate arrays

    def gridSearch():

        def svc():
            startTime = time.time()
            sC = 0
            dC = 5
            sG = 0.00001
            dG = 2.5
            C_range = np.linspace(sC, sC + dC, 150)
            gamma_range = np.linspace(sG, sG + dG, 150)
            param_grid = dict(gamma=gamma_range, C=C_range)
            cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
            grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
            grid.fit(X_train, Y_train)

            print(
                "The best parameters are %s with a score of %0.5f"
                % (grid.best_params_, grid.best_score_)
            )

            print("Tiden var {}".format(time.time() - startTime))

        def mlp():
            startTime = time.time()

            aStart = 1.4446930579460697
            aD = abs(1.6428571428571426 - 1.4446930579460697)*1.1
            a_range = np.linspace(aStart-aD, aStart+aD, 250)
            param_grid = dict(alpha=a_range)
            cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
            grid = GridSearchCV(MLPClassifier(), param_grid=param_grid, cv=cv)
            grid.fit(X_train, Y_train)

            print(
                "The best parameters are %s with a score of %0.5f"
                % (grid.best_params_, grid.best_score_)
            )

            print("Tiden var {}".format(time.time() - startTime))

        mlp()


    def testFunction():

        sumCache = []

        # print("nCV: {}\t\tMean score: {}\tStd: {}\t\tClassifier: {}\t\t\t".format(n,round(np.mean(result), 4), round(np.std(result), 4), clf.__class__.__name__, ))

        n = 10

        for clf in classifiers:
            result = cross_val_score(clf, X_train, Y_train, cv=n)
            print("nCV: {}\t\tMean score: {}\tStd: {}\t\tClassifier: {}\t\t\t".format(n, round(np.mean(result), 4),
                                                                                      round(np.std(result), 4),
                                                                                      clf.__class__.__name__, ))

        estimators = [('svc', SVC()),
                      ('mlp', MLPClassifier(alpha=0.6, max_iter=1000)),
                      ('qda', QuadraticDiscriminantAnalysis())]

        clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=10)
        result = cross_val_score(clf, X_train, Y_train, cv=n)
        print("nCV: {}\t\tMean score: {}\tStd: {}\t\tClassifier: {}\t\t\t".format(n, round(np.mean(result), 4),
                                                                                  round(np.std(result), 4),
                                                                                  clf.__class__.__name__, ))

    #gridSearch()
    testFunction()
