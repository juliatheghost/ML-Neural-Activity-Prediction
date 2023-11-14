#!/usr/bin/env python3

"""
Implementation of SVM ML Model for E-Phys neural analysis.
Part Two. Define technique for main01.py. Select hyper-parameters.

Author: Julia De Geest
Created: June 4th, 2023
"""

from sklearn import svm


class SVMClassifier:

    def __init__(self):
        """ Constructor sets up a sci-kit learn "Support Vector Machines" instance
        :param: self
        """
        C = 0.3  # SVM regularization parameter
        self._internal_classifier = svm.SVC(kernel="linear", C=C, gamma='auto')

    def fit(self, X, y):
        """ This is the method which will be called to train the model.

        :param: X: list of lists. The samples and features which will be used for training.
        :param: y: list. The target/response variable used for training. The data should have the shape:

        :return: self. The underlying model will have some internally saved trained state.
        """
        self._internal_classifier.fit(X, y)

        return self

    def predict(self, X):
        """ Predicts the output targets/responses of a given list of samples. Relies on mechanisms saved after
        train(X, y) was called.

        :param: X: list of lists. The samples and features which will be used for prediction. (Same shape as above).

        :return: y: list. The target/response variables the model decides is optimal for the given samples.
        """
        y = self._internal_classifier.predict(X)

        return y
