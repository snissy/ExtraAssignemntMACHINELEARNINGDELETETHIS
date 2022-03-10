import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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

    # Here we do correlation values. We should plot it first in order go get som insights.  to data

    corrMatrix = trainingData.corr().abs()
    upper = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(bool))

    dropLimit = 0.95
    to_drop = [column for column in upper.columns if any(upper[column] >= dropLimit)]
    # Drop features
    trainingData.drop(to_drop, axis=1, inplace=True)
    evaluationData.drop(to_drop, axis=1, inplace=True)

    # hm = sns.heatmap(corrMatrix, annot=True)
    # hm.set(title="Correlation matrix of IRIS data\n")
    # plt.show()
    # Mean normalize data.
    # mean normalization

    #pca = PCA(n_components=X.shape[1])
    #pca.fit(X)
    #print(pca.explained_variance_ratio_)
    # TODO use sklearn SCALE FUNCTION, StandardScaler
    for cName in evaluationData.columns:
        if evaluationData[cName].dtype.name != 'bool':
            dataMean = evaluationData[cName].mean()
            dataStd = evaluationData[cName].std()
            evaluationData[cName] = (evaluationData[cName] - dataMean) / dataStd
            trainingData[cName] = (trainingData[cName] - dataMean) / dataStd



    # for cName in evaluationData.columns:
    #     if evaluationData[cName].dtype.name != 'bool':
    #         dataMax = evaluationData[cName].max()
    #         dataMin = evaluationData[cName].min()
    #         evaluationData[cName] = (evaluationData[cName] - dataMin) / (dataMax - dataMin)
    #         trainingData[cName] = (trainingData[cName] - dataMin) / (dataMax - dataMin)

    return trainingData, evaluationData


def splitData(data, splitvalue=0.75):
    """
    :param data:
    :param splitvalue:
    :return: training set and validation set.
    """
    return train_test_split(data, test_size=1 - splitvalue)


if __name__ == '__main__':

    trainData, evalData = removeNoise(readData(trainingPath), readData(evaluationPath))
    trainPoint, trainTargets = trainData.drop('y', axis=1), trainData['y']

    # le = preprocessing.LabelEncoder()
    # le.fit(trainTargets)
    # trainTargets = le.transform(trainTargets)

    # Now we can start classify the data

    print(len(trainPoint))

    classifiers = [
        KNeighborsClassifier(15),
        SVC(kernel="linear", C=0.025),
        SVC(),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        MLPClassifier(alpha=0.6, max_iter=1000),
        AdaBoostClassifier(GaussianNB()),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    # trainData, validationData = splitData(trainData)

    # sparse coordinate arrays

    sumCache = []

    #print("nCV: {}\t\tMean score: {}\tStd: {}\t\tClassifier: {}\t\t\t".format(n,round(np.mean(result), 4), round(np.std(result), 4), clf.__class__.__name__, ))

    n = 5
    estimators = [('rf', RandomForestClassifier()), ('mlp', MLPClassifier(alpha=0.6, max_iter=1000)), ('15nn', KNeighborsClassifier(15))]

    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(),  cv=10)
    result = cross_val_score(clf, trainPoint, trainTargets, cv=n)
    print("nCV: {}\t\tMean score: {}\tStd: {}\t\tClassifier: {}\t\t\t".format(n, round(np.mean(result), 4), round(np.std(result), 4), clf.__class__.__name__, ))

    for clf in classifiers:
        result = cross_val_score(clf, trainPoint, trainTargets, cv=n)
        print("nCV: {}\t\tMean score: {}\tStd: {}\t\tClassifier: {}\t\t\t".format(n,round(np.mean(result), 4), round(np.std(result), 4), clf.__class__.__name__, ))



