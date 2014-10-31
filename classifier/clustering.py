#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
===================================
clustering algorithms
===================================
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
# Create Classifiers

classifiers = [("K-Means", cluster.KMeans(n_clusters = num_labels)),
               ("Affinity Propogation", cluster.AffinityPropagation()),
               ("Mean-Shift", cluster.MeanShift()),
               ("Ward Agglomerative Clustering", cluster.AgglomerativeClustering(n_clusters = num_labels)),
               ("DBSCAN", cluster.DBSCAN())]

# ##############################################################################
# Plot result
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

plt.figure(figsize=(17, 9.5))
plt.subplots_adjust(left=.001, right=.999, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)
plot_num = 1
for name, instance in classifiers:
        # predict cluster memberships
        t0 = time.time()
        instance.fit(X)
        t1 = time.time()
        if hasattr(instance, 'labels_'):
            y_pred = instance.labels_.astype(np.int)
        else:
            y_pred = instance.predict(X)

        # Print statistics
        labels = instance.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Metrics for %s' % (name))
        print('-----------------------------------------------')
        print('Estimated number of clusters: %d' % n_clusters_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(y, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(y, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(y, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(y, labels))
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, labels))
        print("")

        # plot
        plt.subplot(1, len(classifiers), plot_num)
        plt.title(name, size = 18)
        plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

        if hasattr(instance, 'cluster_centers_'):
            centers = instance.cluster_centers_
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
