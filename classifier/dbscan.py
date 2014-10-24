#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
===================================
DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""
print(__doc__)

import csv

import numpy as np

from sklearn.cluster import DBSCAN
import sklearn.cross_validation as cross_validation
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import sklearn.cross_validation
import sys

import matplotlib
matplotlib.use('TkAgg') 

##############################################################################
# Constants
NUM_FEATURES = 4

##############################################################################
# Read data
if len(sys.argv) < 2:
    sys.exit("Data File not specified!")

X = np.array([]).reshape(0, NUM_FEATURES)
Y = np.array([])
with open(sys.argv[1], "rb") as inputfile:
    rawdata = csv.reader(inputfile, quoting=csv.QUOTE_NONNUMERIC)
    for i, rawdata_i in enumerate(rawdata):
        length = len(rawdata_i)
        if length < (NUM_FEATURES + 1):
            sys.exit("Row " +
                     str(i) +
                     " has " +
                     str(length) +
                     " elements! Expected " +
                     str(NUM_FEATURES + 1) +
                     " elements")
        X = np.concatenate((X, [ rawdata_i[:NUM_FEATURES] ]))
        Y = np.append(Y, [ rawdata_i[NUM_FEATURES] ])

print(X)
print(Y)

X = StandardScaler().fit_transform(X)

##############################################################################
# Run DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=2)

# Visualization
db = dbscan.fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(Y, labels))
print("Completeness: %0.3f" % metrics.completeness_score(Y, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(Y, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(Y, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(Y, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

##############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
