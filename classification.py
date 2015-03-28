import numpy as np
import csv
import sklearn.linear_model as sklin
import sklearn.ensemble as rf
import sklearn.metrics as skmet
import sklearn.cross_validation as skcv
import sklearn.grid_search as skgs
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
from sklearn.neighbors.dist_metrics import DistanceMetric
from sklearn.multiclass import OneVsOneClassifier


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
            X.append(map(float, row))
    return X


def read_features(X, means,stds, features_fn):
    M = []
    x_rows = len(X)
    i = 1
    for x in X:
        #m = features_fn(x, means, stds)
        M.append(x)
        if i % 100000 == 0:
            print str(i) + ' of ' + str(x_rows) + ' rows processed...'
        i += 1
    print str(i) + ' of ' + str(x_rows) + ' rows processed...'
    poly = preprocessing.PolynomialFeatures(3)
    M = np.matrix(M)
    Xs = M[:,0:9]
    Xs = poly.fit_transform(Xs)
    M = np.concatenate((M,Xs),axis=1)
    return M


def some_features(x):
    return [x]
#    return [np.log(1 + np.abs(x)), np.exp(x), x, x ** 2, np.abs(x)]


# Assume that all values in x are ready-to-use features (i. e. no timestamps)
def simple_implementation(x, means,stds):
#    x = map(float, x)
    fs = [y for i in range(8) for y in some_features((float(x[i])-means[i])/stds[i])]
    fs.extend(x[9:])
    fs.append(1)
    return fs


def ortho(fns, x, means,stds):
    y = []
    for fn in fns:
        y.extend(fn(x, means,stds))
    return y


def predict_and_print(name, class1, class2, X):
    Ypred = np.array([class1.predict(X), class2.predict(X)])
    np.savetxt('project_data/' + name + '.txt', Ypred.T, fmt='%i', delimiter=',')


def lin_classifier(Xtrain, Ytrain):
    classifier = LinearSVC()
    classifier.fit(Xtrain, Ytrain)
    print 'LIN: coef', classifier.coef_
    return classifier

def onevsone_classifier(Xtrain,Ytrain):
    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
    classifier.fit(Xtrain,Ytrain)
    return classifier

def tree_classifier(Xtrain, Ytrain):
    '''param_grid = {
                  'max_features': range(20,53,15),}
        #'min_samples_split': range(2,4,1),
        #'max_depth': range(10,50,20)
    classifier = ExtraTreesClassifier(n_jobs=-1,verbose=0, min_samples_leaf=3, n_estimators = 220, criterion = 'entropy')

    classifier.fit(Xtrain, Ytrain)
    print 'TREE: classifier: ', classifier
    print 'TREE: classifier.score: ', score(Ytrain, classifier.predict(Xtrain))
    scorefun = skmet.make_scorer(lambda x, y: -score(x, y))
    grid_search = skgs.GridSearchCV(classifier, param_grid, scoring=scorefun, cv=5)
    grid_search.fit(Xtrain, Ytrain)
    print 'TREE: best_estimator_: ', grid_search.best_estimator_
    print 'TREE: best_estimator_.score: ', score(Ytrain, grid_search.predict(Xtrain))
    return grid_search.best_estimator_

    '''

    #evaluated classifier
    classifier = ExtraTreesClassifier(n_jobs=-1,
                                      criterion='entropy',
                                      max_features=108,
                                      min_samples_split=4,
                                      n_estimators=421)
    classifier.fit(Xtrain,Ytrain)
    return classifier


def forest_classifier(Xtrain, Ytrain):
    param_grid = {'n_estimators': range(1, 100, 25), 'max_features': range(1, 50, 5), 'min_samples_split': range(1,10,1), 'max_depth': range(1,1000,100)}
    classifier = RandomForestClassifier(n_jobs=-1,verbose=1, max_depth=None,min_samples_split=1)
    classifier.fit(Xtrain, Ytrain)
    print 'FOREST: classifier: ', classifier
    print 'FOREST: classifier.score: ', score(Ytrain, classifier.predict(Xtrain))
    scorefun = skmet.make_scorer(lambda x, y: -score(x, y))
    grid_search = skgs.GridSearchCV(classifier, param_grid, scoring=scorefun, cv=3)
    grid_search.fit(Xtrain, Ytrain)
    print 'FOREST: best_estimator_: ', grid_search.best_estimator_
    print 'FOREST: best_estimator_.score: ', score(Ytrain, grid_search.predict(Xtrain))
    return grid_search.best_estimator_


def knn_classifier(Xtrain, Ytrain):
    param_grid = {'n_neighbors': [4, 5, 8, 16], 'weights': ['uniform'],
#                  'metric': map(DistanceMetric.get_metric, ['manhatten', 'jaccard'])
    }
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


def svm_classifier(Xtrain, Ytrain):
    #param_grid = {'weights': ['uniform']}      #standard assumption by svm
    C_range = np.logspace(-3,3,11)
    gamma_range = np.logspace(-3,3,11)
    degree_range = range(2,4,1)
    coef0_range = np.logspace(-3,3,11)
    cv = StratifiedKFold(y=Ytrain, n_folds=3)
    param_grid = dict(gamma = gamma_range, C = C_range, degree = degree_range, coef0 = coef0_range)
    grid = GridSearchCV(svm.SVC(verbose=True, kernel = 'poly'), param_grid = param_grid, cv=cv, verbose=5, n_jobs=-1)
    #if opt == 0:
    #    classifier = svm.SVC(gamma = 0.063095734448019303, C= 15.848931924611142, degree = 3, verbose=True)
    #else:
    #    classifier = svm.SVC(gamma=0.25118864315095796,C=3.9810717055349691, degree = 3, verbose=True)
    #classifier.fit(Xtrain,Ytrain)
    grid.fit(Xtrain,Ytrain)
    print 'The best classifier is: %s' %grid.best_estimator_

    return grid.best_estimator_
    #return classifier


def multi_classifier(classifiers):
    return {}




def regress(fn, name, X, Y, Xval, Xtestsub):
    Xtrain, Xtest, Ytrain, Ytest = skcv.train_test_split(X, Y, train_size=0.7)

    class1 = fn(Xtrain, Ytrain[:, 0])
    class2 = fn(Xtrain, Ytrain[:, 1])
    print 'DEBUG: classifier trained'

    print 'SCORE:', name, ' - trainset ', sumscore_classifier(class1, class2, Xtrain, Ytrain)
    print 'SCORE:', name, ' - test ', sumscore_classifier(class1, class2, Xtest, Ytest)

    predict_and_print('validate_y_' + name, class1, class2, Xval)
    predict_and_print('test_y_' + name, class1, class2, Xtestsub)



def regress_no_split(fn, name, X, Y, Xval, Xtestsub):
    class1 = fn(X, Y[:, 0])
    class2 = fn(X, Y[:, 1])

    print 'SCORE:', name, ' - all ', sumscore_classifier(class1, class2, X, Y)
    ''' score_fn = skmet.make_scorer(score)
    scores = skcv.cross_val_score(class1, X, Y[:, 0], scoring=score_fn, cv=3)
    print 'SCORE:', name, ' - (cv) mean on 1 : ', np.mean(scores), ' +/- ', np.std(scores)
    scores = skcv.cross_val_score(class2, X, Y[:, 1], scoring=score_fn, cv=3)
    print 'SCORE:', name, ' - (cv) mean on 2 : ', np.mean(scores), ' +/- ', np.std(scores)
    '''
    predict_and_print('validate_y_' + name, class1, class2, Xval)
    predict_and_print('test_y_' + name, class1, class2, Xtestsub)


def read_and_regress(feature_fn):
    Xo = read_path('project_data/train.csv')
    print 'data points: ', len(Xo)

    XM = np.matrix(Xo)
    means = [np.mean(XM[:,i]) for i in range(np.shape(XM)[1])]
    #print 'means: ', means
    stds = [np.std(XM[:,i]) for i in range(np.shape(XM)[1])]
    #print 'stds: ', stds

    Y = np.genfromtxt('project_data/train_y.csv', delimiter=',')
    #np.savetxt('project_data/' + 'DEBUG_X' + '.txt', Xo, fmt='%f', delimiter=',')

    X = read_features(Xo, means, stds, feature_fn)

    #X[:,0:9] = preprocessing.scale(X[:,0:9])
    #np.savetxt('project_data/' + 'DEBUG_Xscaled' + '.txt', X, fmt='%1.2f', delimiter=',')

    print 'DEBUG: total nb of base-functions: %d' % np.shape(X)[1]
    Xvalo = read_path('project_data/validate.csv')
    Xtesto = read_path('project_data/test.csv')
    Xval = read_features(Xvalo, means, stds, feature_fn)
    Xtest = read_features(Xtesto, means, stds, feature_fn)
    #Xval[:,0:9] = preprocessing.scale(Xval[:,0:9])      #only scale features that are not 'one-hot-encoded'
#   Xtest = preprocessing.scale(Xtest)
    # For now, we don't need to generate test
    #Xtest = Xval
    print 'DEBUG: read in everything'

    #regress(lin_classifier, 'lin', X, Y, Xval, Xtest)
    #regress(knn_classifier, 'knn', X, Y, Xval, Xtest)
    regress_no_split(tree_classifier, 'tree', X, Y, Xval, Xtest)
    #regress(onevsone_classifier, 'onevsone', X, Y, Xval, Xtest)
    #regress_no_split(forest_classifier, 'forest', X, Y, Xval, Xtest)
    #regress_no_split(svm_classifier, 'svm', X, Y, Xval, Xtest)
    #regress(svm_classifier(0),'svm',X,Y,Xval,Xtest)
    '''Yval = np.genfromtxt('project_data/validate_y_tree.txt', delimiter=',')
    print 'training classifier on predicted data'
    regress(tree_classifier,'tree',Xval,Yval,Xval,Xtest)'''


if __name__ == "__main__":
    read_and_regress(lambda x, means, stds: ortho([simple_implementation], x, means, stds))
