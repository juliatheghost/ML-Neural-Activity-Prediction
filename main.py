#!/usr/bin/env python3

"""
BI 410L Final Project. Implementation of SVM ML Model for E-Phys neural analysis.
Part Three. Implement defined ML algorithm and get accuracy.

Author: Julia De Geest
Created: June 4th, 2023
"""

import pandas as pd
from SVM_neural import SVMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class Experiment01:

    @staticmethod
    def run():
        """ Load the data, set up the machine learning model, train the model, get predictions from the model based on
        unseen data, assess the accuracy of the model, and print the results.

        :return: test_y, orientation_predictions. Runs the Model and prints report!
        """
        train_X, train_y, test_X, test_y = Experiment01.load_data()

        EPhys_ML = SVMClassifier()

        EPhys_ML.fit(X=train_X, y=train_y)
        orientation_predictions = EPhys_ML.predict(X=test_X)

        target_names = ['angle 1', 'angle 2', 'angle 3', 'angle 4']

        print(classification_report(test_y, orientation_predictions, target_names=target_names, zero_division=1))

        return test_y, orientation_predictions

    @staticmethod
    def load_data():
        """ Load the data and partitions it into testing and training data.
        :param: None. Relies on the pre-generated file data.csv

        :return: train_X, train_y, test_X, test_y: each as list.
        """
        # Clean data using the pre-generated data.csv file
        clean_data = pd.read_csv('data.csv')

        X = clean_data.iloc[:, :-1]
        y = clean_data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.07, random_state=42)  # test size is small
        # due to data restrictions

        # Normalize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, y_train, X_test, y_test

    @staticmethod
    def _get_accuracy(pred_y, true_y):
        """ Calculates the overall percentage accuracy.
        :param: pred_y: predicted values.
        :param: true_y: ground truth values.

        :return: The accuracy, formatted as a number in [0, 1].
        """
        if len(pred_y) != len(true_y):
            raise Exception("Different number of prediction-values than truth-values.")

        number_of_agreements = 0
        number_of_pairs = len(true_y)

        for individual_prediction_value, individual_truth_value in zip(pred_y, true_y):
            if individual_prediction_value == individual_truth_value:
                number_of_agreements += 1

        accuracy = number_of_agreements / number_of_pairs

        return accuracy


if __name__ == "__main__":
    # Run experiment. Prints a classification report from sklearn.metrics and an overall accuracy from _get_accuracy.
    y_true, y_pred = Experiment01.run()
    accuracy = Experiment01._get_accuracy(y_true, y_pred)

    print("Overall accuracy is: %s" % accuracy)