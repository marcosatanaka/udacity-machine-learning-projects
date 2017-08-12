import glob

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import feature as feature


def run():
    x_all, y_all = get_chord_paths_and_labels()

    x_all = feature.get_features(x_all)
    cross_validation_score(x_all, y_all, 9)


def get_chord_paths_and_labels():
    x = []
    y = []

    for file_name in glob.glob1("data", "*.wav"):
        x.append("data/" + file_name)
        y.append(file_name[0])

    return x, y


def cross_validation_score(x_all, y_all, n_cv):
    clf = SVC(random_state=1)
    parameters = [
        {'kernel': ['linear'], 'C': np.arange(0.1, 1, 0.1)},
        {'kernel': ['rbf'], 'C': np.arange(0.1, 1, 0.1), 'gamma': np.arange(0.1, 1, 0.1)},
        {'kernel': ['sigmoid'], 'C': np.arange(0.1, 1, 0.1), 'gamma': np.arange(0.1, 1, 0.1),
         'coef0': [0, 1, 2, 3, 4, 5]}
    ]
    clf = best_estimator(clf, parameters, x_all, y_all)

    scores = cross_val_score(clf, x_all, y_all, cv=n_cv, scoring="accuracy")

    print "Acuracia SVC ({}-Fold): {:.2f}\n".format(n_cv, scores.mean())

    scores = cross_val_score(GaussianNB(), x_all, y_all, cv=n_cv, scoring="accuracy")
    print "Acuracia GNB ({}-Fold): {:.2f}\n".format(n_cv, scores.mean())

    clf = DecisionTreeClassifier(random_state=1)
    parameters = [
        {'max_depth': np.arange(2, 10, 1)}
    ]
    clf = best_estimator(clf, parameters, x_all, y_all)
    scores = cross_val_score(clf, x_all, y_all, cv=n_cv, scoring="accuracy")
    print "Acuracia DTC ({}-Fold): {:.2f}\n".format(n_cv, scores.mean())

    scores = cross_val_score(AdaBoostClassifier(random_state=1), x_all, y_all, cv=n_cv, scoring="accuracy")
    print "Acuracia ABC ({}-Fold): {:.2f}\n".format(n_cv, scores.mean())


def best_estimator(clf, parameters, x_train, y_train):
    scorer = make_scorer(accuracy_score)

    grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

    grid_obj = grid_obj.fit(x_train, y_train)

    print "Melhores parametros: {}".format(grid_obj.best_params_)

    return grid_obj.best_estimator_


if __name__ == '__main__':
    run()
