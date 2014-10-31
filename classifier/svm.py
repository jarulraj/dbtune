#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
===================================
SVM
===================================
"""
print(__doc__)

from collections import Counter
import csv

import numpy as np

from sklearn import svm
import sklearn.cross_validation as cross_validation
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import sklearn.cross_validation
import sys
import time

# CONFIGURATION

LABEL_FIELD = 9

##############################################################################
# Constants
NUM_FOLDS = 5

##############################################################################
# Read data
if len(sys.argv) < 2:
    sys.exit("Data File not specified!")

X = np.array([])
y = np.array([])
num_features = None
with open(sys.argv[1], "rb") as inputfile:
    text_feature_map = {}
    next_text_feature_number = {}
    rawdata = csv.reader(inputfile)

    # Convert data
    for i, rawdata_i in enumerate(rawdata):
        length = len(rawdata_i)

        # Get number of features if first row
        if i == 0:
            num_features = length - 1
            if num_features < 1:
                sys.exit("Need at least one feature and exactly one label!")

            print("Detected " + str(num_features) + " features")
            X = X.reshape(0, num_features)

        # Check row length
        if length != (num_features + 1):
            sys.exit("Row " +
                     str(i + 1) +
                     " has " +
                     str(length) +
                     " elements! Expected " +
                     str(num_features + 1) +
                     " elements")

        # Convert row to numbers
        converted_row = np.array([])
        for column, entry in enumerate(rawdata_i):
            converted_value = None
            try:
                converted_value = float(entry)
            except ValueError:
                converted_value = None

            if converted_value is None:
                if not (column in text_feature_map):
                    if i == 0:
                        text_feature_map[column] = {}
                        next_text_feature_number[column] = 0
                    else:
                        sys.exit("Encountered text feature \"" +
                                 entry +
                                 "\" in row " +
                                 str(i + 1) +
                                 " for numerical column " +
                                 str(column) +
                                 "!")

                if entry in text_feature_map[column]:
                    converted_value = text_feature_map[column][entry]
                else:
                    text_feature_map[column][entry] = next_text_feature_number[column]
                    converted_value = next_text_feature_number[column]
                    next_text_feature_number[column] += 1

            converted_row = np.append(converted_row, [ converted_value ])

        X = np.concatenate( (X, [ np.append(converted_row[:LABEL_FIELD], converted_row[LABEL_FIELD + 1:]) ]))
        y = np.append(y, [ converted_row[LABEL_FIELD] ])

X = StandardScaler().fit_transform(X)

# Count number of labels
y_counter = Counter()
for label in y:
    y_counter[label] += 1
num_labels = len(y_counter.keys())

print("Detected " + str(num_labels) + " labels")
print("")

##############################################################################
# SVM
clf = svm.NuSVC()
scores = cross_validation.cross_val_score(clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
