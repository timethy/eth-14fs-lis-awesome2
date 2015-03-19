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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors.dist_metrics import DistanceMetric


def score(gtruth, gpred):
    # Minimizing this should result in minimizing sumscore in the end.
    # We do not actually need the len(gtruth), but it enhances debugging, since it then corresponds to the sumscore.
    return float(np.sum(gtruth != gpred))/(len(gtruth))


def sumscore(gtruth1, gtruth2, gpred1, gpred2):
    return float((np.sum(gtruth1 != gpred1) + np.sum(gtruth2 != gpred2)))/(2*len(gtruth1))


def sumscore_classifier(class1, class2, X, Y):
    return sumscore(Y[:, 0], Y[:, 1], class1.predict(X), class2.predict(X))


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
    x = map(float, x)
    x.append(x[0]*x[1])
    x.append(x[5]*x[6])
#    x[3] = np.abs(x[3])
    return x


def ortho(fns, x):
    y = []
    for fn in fns:
        y.extend(fn(x))
    return np.array(y)


def predict_and_print(name, class1, class2, X):
    Ypred = np.array([class1.predict(X), class2.predict(X)])
    np.savetxt('project_data/' + name + '.txt', Ypred.T, fmt='%i', delimiter=',')


def lin_classifier(Xtrain, Ytrain):
    classifier = LinearSVC()
    classifier.fit(Xtrain, Ytrain)
    return classifier


def tree_classifier(Xtrain, Ytrain):
    param_grid = {'n_estimators': range(5, 51, 5), 'max_depth': range(10, 101, 10)}
    classifier = RandomForestClassifier(n_jobs=4)
    classifier.fit(Xtrain, Ytrain)
    print 'TREE: classifier: ', classifier
    print 'TREE: classifier.score: ', score(Ytrain, classifier.predict(Xtrain))
    scorefun = skmet.make_scorer(lambda x, y: -score(x, y))
    grid_search = skgs.GridSearchCV(classifier, param_grid, scoring=scorefun, cv=5)
    grid_search.fit(Xtrain, Ytrain)
    print 'TREE: best_estimator_: ', grid_search.best_estimator_
    print 'TREE: best_estimator_.score: ', score(Ytrain, grid_search.predict(Xtrain))
    return grid_search.best_estimator_


def knn_classifier(Xtrain, Ytrain):
    param_grid = {'n_neighbors': [4, 8, 16], 'weights': ['uniform', 'distance']}
    classifier = KNeighborsClassifier(algorithm='auto')
    classifier.fit(Xtrain, Ytrain)
    print 'KNN: classifier: ', classifier
    print 'KNN: classifier.score: ', score(Ytrain, classifier.predict(Xtrain))
    scorefun = skmet.make_scorer(lambda x, y: -score(x, y))
    grid_search = skgs.GridSearchCV(classifier, param_grid, scoring=scorefun, cv=5)
    grid_search.fit(Xtrain, Ytrain)
    print 'KNN: best_estimator_: ', grid_search.best_estimator_
    print 'KNN: best_estimator_.score: ', score(Ytrain, grid_search.predict(Xtrain))
    return grid_search.best_estimator_


def regress(fn, name, X, Y, Xval, Xtestsub):
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.8)

    class1 = fn(Xtrain, Ytrain[:, 0])
    class2 = fn(Xtrain, Ytrain[:, 1])

    print 'SCORE:', name, ' - trainset ', sumscore_classifier(class1, class2, Xtrain, Ytrain)
    print 'SCORE:', name, ' - test ', sumscore_classifier(class1, class2, Xtest, Ytest)

    predict_and_print('validate_y_' + name, class1, class2, Xval)
    predict_and_print('test_y_' + name, class1, class2, Xtestsub)


def regress_no_split(fn, name, X, Y, Xval, Xtestsub):
    class1 = fn(X, Y[:, 0])
    class2 = fn(X, Y[:, 1])

    print 'SCORE:', name, ' - all ', sumscore_classifier(class1, class2, X, Y)

    score_fn = skmet.make_scorer(score)
    scores = skcv.cross_val_score(class1, X, Y[:, 0], scoring=score_fn, cv=5)
    print 'SCORE:', name, ' - (cv) mean : ', np.mean(scores), ' +/- ', np.std(scores)
    scores = skcv.cross_val_score(class2, X, Y[:, 1], scoring=score_fn, cv=5)
    print 'SCORE:', name, ' - (cv) mean : ', np.mean(scores), ' +/- ', np.std(scores)

    predict_and_print('validate_y_' + name, class1, class2, Xval)
    predict_and_print('test_y_' + name, class1, class2, Xtestsub)


def read_and_regress(feature_fn):
    Xo = read_path('project_data/train.csv')
    print 'data points: ', len(Xo)

    Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')
    X = read_features(Xo, feature_fn)
    print 'DEBUG: total nb of base-functions: %d' % np.shape(X)[1]
    Xvalo = read_path('project_data/validate.csv')
    Xtesto = read_path('project_data/test.csv')
    Xval = read_features(Xvalo, feature_fn)
    Xtest = read_features(Xtesto, feature_fn)
    print 'DEBUG: read in everything'

    regress(lin_classifier, 'lin', X, Y, Xval, Xtest)
    regress(knn_classifier, 'knn', X, Y, Xval, Xtest)
    regress_no_split(tree_classifier, 'tree', X, Y, Xval, Xtest)


if __name__ == "__main__":
    read_and_regress(lambda x: ortho([simple_implementation], x))
