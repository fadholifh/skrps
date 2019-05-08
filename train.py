# Import required libraries


import sklearn.metrics
import sen

import csv
import os
import re
import nltk
import scipy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn import model_selection


# Generating the Training and testing vectors

def getTrainingAndTestData():
    X = []
    y = []

    # # Training data 1: Sentiment 140
    # f = open(r'training_test.csv', 'r', encoding='ISO-8859-1')
    # reader = csv.reader(f)
    #
    # for row in reader:
    #     X.append(row[5])
    #     y.append(1 if (row[0] == '4') else 0)

    # Training data 3: Umich

    f = open(r'cpa.txt', 'r', encoding='ISO-8859-1')
    reader = csv.reader(f)

    for row in reader:
        line = ' '.join(row)
        lyst = line.split('\t')
        X.append(lyst[1])
        y.append(int(lyst[0]))

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test


# Process Tweets (Stemming+Pre-processing)

def processTweets(X_train, X_test):
    X_train = [sen.stemm(sen.stopw(sen.cleaning(data))) for data in X_train]
    X_test = [sen.stemm(sen.stopw(sen.cleaning(data))) for data in X_test]
    return X_train, X_test


# SVM classifier

def classifier(X_train, y_train):
    vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))
    svm_clf = svm.LinearSVC(C=0.1)
    vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
    vec_clf.fit(X_train, y_train)
    joblib.dump(vec_clf, 'svmClassifierss.pkl', compress=3)
    return vec_clf


# Main function

def main():
    X_train, X_test, y_train, y_test = getTrainingAndTestData()
    X_train, X_test = processTweets(X_train, X_test)
    vec_clf = classifier(X_train, y_train)
    y_pred = vec_clf.predict(X_test)
    print(sklearn.metrics.classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
