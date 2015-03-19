import numpy as np                  #basic linear algebra
import csv
import sklearn.linear_model as sklin
import sklearn.ensemble as rf
import sklearn.metrics as skmet
import sklearn.cross_validation as skcv
import sklearn.grid_search as skgs
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.dist_metrics import DistanceMetric


def score(gtruth, gpred):
    # Minimizing this should result in minimizing sumscore in the end.
    # We do not actually need the len(gtruth), but it enhances debugging, since it then corresponds to the sumscore.
    return float(np.sum(gtruth != gpred))/(len(gtruth))


def sumscore(gtruth1, gtruth2, gpred1, gpred2):
    return float((np.sum(gtruth1 != gpred1) + np.sum(gtruth2 != gpred2)))/(2*len(gtruth1))


def sumscore_classifier(class1, class2, X, Y):
    return sumscore(Y[:, 0], Y[:, 1], class1.predict(X), class2.predict(X))


def indicators(vals, x):
    y = []
    for val in vals:
        if int(x)== int(val):
            y.append(1)
        else:
            y.append(-1)
    return y


def read_path(inpath):
    X = []
    with open(inpath, 'r') as fin:
        reader = csv.reader(fin, delimiter=',')
        for row in reader:
            X.append(row)
    return X


def read_features(X, features_fn):
    M = []
    x_rows = len(X)
    i = 1
    for x in X:
        m = features_fn(x)
        M.append(m)
        if i % 100000 == 0:
            print str(i) + ' of ' + str(x_rows) + ' rows processed...'
        i += 1
    return np.matrix(M)


# Assume that all values in x are ready-to-use features (i. e. no timestamps)
def simple_implementation(x):
    return x


def ortho(fns, x):
    y = []
    for fn in fns:
        y.extend(fn(x))
    return np.array(y)


def predict_and_print(name, class1, class2, X):
    Ypred = np.array([class1.predict(X), class2.predict(X)])
    np.savetxt('project_data/' + name + '.txt', Ypred.T, fmt='%i', delimiter=',')


def knn_classifier(Xtrain, Ytrain):
    param_grid = {'n_neighbors': range(1, 15), 'weights': ['uniform', 'distance']}
    classifier = KNeighborsClassifier(algorithm='auto')
    classifier.fit(Xtrain, Ytrain)
    print 'classifier.score: ', score(Ytrain, classifier.predict(Xtrain))
    scorefun = skmet.make_scorer(lambda x, y: -score(x, y))
    grid_search = skgs.GridSearchCV(classifier, param_grid, scoring=scorefun, cv=5)
    grid_search.fit(Xtrain, Ytrain)
    print 'grid_search.best_estimator_: ', grid_search.best_estimator_
    return grid_search.best_estimator_


def regress_knn(X, Y, Xval, Xtest):
    # always split training and test data!
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.8)
    print 'DEBUG: data split up into train and test data'

    class1 = knn_classifier(Xtrain, Ytrain[:, 0])
    class2 = knn_classifier(Xtrain, Ytrain[:, 1])

    print 'score on trainset: ', sumscore_classifier(class1, class2, Xtrain, Ytrain)
    print 'score on test: ', sumscore_classifier(class1, class2, Xtest, Ytest)

    predict_and_print('validate_y_knn', class1, class2, Xval)
    predict_and_print('test_y_knn', class1, class2, Xtest)

def read_and_regress(feature_fn):
    Xo = read_path('project_data/train.csv')
    print 'data points: ', len(Xo)

    Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')
    print 'DEBUG: data read'
    X = read_features(Xo, feature_fn)
    print 'DEBUG: total nb of base-functions: %d' % np.shape(X)[1]
    print 'DEBUG: transform training data features'
    Xvalo = read_path('project_data/validate.csv')
    Xtesto = read_path('project_data/test.csv')
    print 'DEBUG: transform validation data features'
    Xval = read_features(Xvalo, feature_fn)
    Xtest = read_features(Xtesto, feature_fn)
    print 'DEBUG: features transformed'

    regress_knn(X, Y, Xval, Xtest)


if __name__ == "__main__":
    read_and_regress(lambda x: ortho([simple_implementation], x))
