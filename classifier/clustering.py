#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
===================================
DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""
print(__doc__)

from collections import Counter
import csv

import numpy as np

import sklearn.cluster as cluster
import sklearn.cross_validation as cross_validation
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import sklearn.cross_validation
import sys
import time

##############################################################################
# Read data
if len(sys.argv) < 2:
    sys.exit("Data File not specified!")

X = np.array([])
Y = np.array([])
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

        X = np.concatenate((X, [ converted_row[:num_features] ]))
        Y = np.append(Y, [ converted_row[num_features] ])

X = StandardScaler().fit_transform(X)

# Count number of labels
Y_counter = Counter()
for y in Y:
    Y_counter[y] += 1
num_labels = len(Y_counter.keys())

print("Detected " + str(num_labels) + " labels")

##############################################################################
# Create Classifiers

classifiers = [("K-Means", cluster.KMeans(n_clusters = num_labels)),
               ("Affinity Propogation", cluster.AffinityPropagation()),
               ("Mean-Shift", cluster.MeanShift()),
               ("Ward Agglomerative Clustering", cluster.AgglomerativeClustering(n_clusters = num_labels)),
               ("DBSCAN", cluster.DBSCAN())]

# ##############################################################################
# # Plot result
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

plt.figure(figsize=(17, 9.5))
plt.subplots_adjust(left=.001, right=.999, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)
plot_num = 1
for name, object in classifiers:
        # predict cluster memberships
        t0 = time.time()
        object.fit(X)
        t1 = time.time()
        if hasattr(object, 'labels_'):
            y_pred = object.labels_.astype(np.int)
        else:
            y_pred = object.predict(X)

        # plot
        plt.subplot(1, len(classifiers), plot_num)
        plt.title(name, size = 18)
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

        if hasattr(object, 'cluster_centers_'):
            centers = object.cluster_centers_
            center_colors = colors[:len(centers)]
            plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()
