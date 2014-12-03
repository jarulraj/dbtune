#! /usr/bin/env python

import os
import subprocess
import logging
import datetime
import argparse
import glob
import pprint
import sys
import re
import fnmatch
import string
import random
import time
import csv
import numpy as np
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydot

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import sklearn.cross_validation
import sklearn.cross_validation as cross_validation

import sklearn.cluster as cluster
from sklearn import svm
from sklearn import tree
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn import gaussian_process

# # LOGGING CONFIGURATION
LOG = logging.getLogger(__name__)
LOG_handler = logging.StreamHandler()
LOG_formatter = logging.Formatter(
    fmt='%(asctime)s [%(funcName)s:%(lineno)03d] %(levelname)-5s: %(message)s',
    datefmt='%m-%d-%Y %H:%M:%S'
)
LOG_handler.setFormatter(LOG_formatter)
LOG.addHandler(LOG_handler)
LOG.setLevel(logging.INFO)
np.set_printoptions(suppress=True)

# # CONFIGURATION
BASE_DIR = os.path.dirname(__file__)

BENCHMARK_FIELD = 0
LABEL_FIELD = 0
BENCHMARK_LABEL_FIELD = LABEL_FIELD
THROUGHPUT_LABEL_FIELD = 1
NUM_FOLDS = 5

feature_list = []
feature_name_only_list = []
benchmark_list = []

# Get info
def get_info(feature_info):
    # FEATURE LIST
    if feature_info:
        print(feature_list)

    # BENCHMARK LIST
    print(benchmark_list)


# Preprocess feature data
def preprocess(filename, normalize_data, label_field):
    X = np.array([])
    y = np.array([])
    num_features = None
    global feature_list
    global benchmark_list

    print("LABEL FIELD : " + str(label_field))

    with open(filename, "rb") as inputfile:
        text_feature_map = {}
        next_text_feature_number = {}
        rawdata = csv.reader(inputfile)

        # Convert data
        for i, rawdata_i in enumerate(rawdata):
            length = len(rawdata_i)

            if i == 0:
                num_features = length - 1
                f_list = rawdata_i

                if num_features < 1:
                    sys.exit("Need at least one feature and exactly one label!")

                print("Detected " + str(num_features) + " features")
                X = X.reshape(0, num_features)

                for index, item in enumerate(f_list):
                    feature_list.append((index, item))
                    feature_name_only_list.append(item)

                continue

            # Check row length
            if length != (num_features + 1):
                sys.exit("Row " + str(i + 1) + " has " + str(length) + " elements! Expected " +
                         str(num_features + 1) + " elements")

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
                        if i == 1:
                            text_feature_map[column] = {}
                            next_text_feature_number[column] = 0
                        else:
                            print(X[:i-1,column])
                            sys.exit("Encountered text feature \"" + entry + "\" in row " +
                                     str(i + 1) + " for numerical column " + feature_list[column] + "!")

                    if entry in text_feature_map[column]:
                        converted_value = text_feature_map[column][entry]
                    else:
                        text_feature_map[column][entry] = next_text_feature_number[column]
                        converted_value = next_text_feature_number[column]
                        next_text_feature_number[column] += 1

                converted_row = np.append(converted_row, [ converted_value ])

            X = np.concatenate( (X, [ np.append(converted_row[:label_field], converted_row[label_field + 1:]) ]))
            y = np.append(y, [ converted_row[label_field] ])

    # Normalize features
    if normalize_data:
        X = StandardScaler().fit_transform(X)

    # Count number of labels
    y_counter = Counter()
    for label in y:
        y_counter[label] += 1
    num_labels = len(y_counter.keys())

    # BENCHMARK LIST
    benchmark_list = text_feature_map[BENCHMARK_FIELD]
    benchmark_list = sorted(benchmark_list.items(), key=lambda x:x[1])

    print("Detected " + str(num_labels) + " labels")
    inputfile.close()

    return (X, y, num_labels)


# Splitting helper
def split_data(X, y, ratio):
    num_samples = len(X)/ratio

    X_train = X[num_samples:, :]
    y_train = y[num_samples:]
    X_test = X[:num_samples, :]
    y_test = y[:num_samples]

    return (X_train, y_train, X_test, y_test)

# Clustering
def clustering_classifier(X, y, num_labels):

    classifiers = [("K-Means", cluster.KMeans(n_clusters = num_labels)),
                   ("Affinity Propogation", cluster.AffinityPropagation()),
                   ("Mean-Shift", cluster.MeanShift()),
                   ("Ward Agglomerative Clustering", cluster.AgglomerativeClustering(n_clusters = num_labels))]

    # Plot
    plt.figure(figsize=(30, 6.5))
    plt.subplots_adjust(left=.001, right=.999, bottom=.001, top=.96, wspace=.05, hspace=.01)

    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    plot_num = 1

    for name, instance in classifiers:
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

            print(labels)

            print("Silhouette Coefficient: %0.3f"
                  % metrics.silhouette_score(X, labels))
            print("")

            # plot
            plt.subplot(1, len(classifiers), plot_num)
            plt.title(name, size = 18)
            plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=20)

            if hasattr(instance, 'cluster_centers_'):
                centers = instance.cluster_centers_
                center_colors = colors[:len(centers)]
                plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.xticks(())
            plt.yticks(())
            #plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
            #         transform=plt.gca().transAxes, size=15,
            #         horizontalalignment='right')
            plot_num += 1

    plt.savefig('clustering.pdf', format='pdf', dpi=1000)


# SVM
def svm_classifier(X, y):

    clf = svm.SVC()

    [X_train, y_train, X_test, y_test] = split_data(X, y, 2)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))

    scores = cross_validation.cross_val_score(clf, X, y, cv=2, scoring='precision')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Decision trees
def decision_tree_classifier(X, y, depth, leaf_nodes, output_file_name):
    # Set depth and leaf nodes
    clf = tree.DecisionTreeClassifier(max_depth = depth, max_leaf_nodes= leaf_nodes)

    [X_train, y_train, X_test, y_test] = split_data(X, y, 2)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics_data = metrics.classification_report(y_test, y_pred)
    print(metrics_data)

    scores = cross_validation.cross_val_score(clf, X, y, cv=2, scoring='precision')
    accuracy_data = "Accuracy: %0.2f %0.2f \n" % (scores.mean(), scores.std() * 2)
    print(accuracy_data)

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_name_only_list[1:])
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(output_file_name)

    return (metrics_data, accuracy_data)

# LASSO
def lasso_estimator(X, y):
    alpha = 0.1
    clf = linear_model.Lasso(alpha = alpha)
    #clf = make_pipeline(Normalizer(norm="l2"), linear_model.Lasso(alpha = alpha, max_iter=100))

    [X_train, y_train, X_test, y_test] = split_data(X, y, 2)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(y_test)
    print(y_pred)
    print(clf.sparse_coef_)
    print(r2_score(y_test, y_pred))

# GP
def gp_estimator(X, y):
    clf = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
    #clf = gaussian_process.GaussianProcess(regr='linear')

    [X_train, y_train, X_test, y_test] = split_data(X, y, 2)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(y_test)
    print(y_pred)
    print(r2_score(y_test, y_pred))

def estimate_performance(file):
    methods = [("Lasso Regression", linear_model.Lasso(alpha = 0.05, max_iter = 100000)),
               ("Gaussian Processes", gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1))]

    [_, y_benchmark, num_benchmarks] =  preprocess(file, normalize_data, BENCHMARK_LABEL_FIELD)
    [X_throughput_combined, y_throughput_combined, num_throughputs] = preprocess(file, normalize_data, THROUGHPUT_LABEL_FIELD)

    for name, instance in methods:
        print("===========================================================================")
        print("Using method %s" % name)
        print("===========================================================================")

        for benchmark_number in range(num_benchmarks):
            print("-----------------------")
            print("Estimating for benchmark %s" % benchmark_list[benchmark_number][0])
            print("-----------------------")

            sample_filter = y_benchmark == benchmark_number
            X_throughput = X_throughput_combined[sample_filter, :]
            y_throughput = y_throughput_combined[sample_filter, :]

            print("Found %d samples, doing two-way CV" % X_throughput.shape[0])

            [X_train, y_train, X_test, y_test] = split_data(X_throughput, y_throughput, 2)
            instance.fit(X_train, y_train)
            y_pred = instance.predict(X_test)

            np.set_printoptions(suppress=True)
            print(y_test[:20])
            print(y_pred[:20])

            print("Estimator got R2 Score %f" % r2_score(y_test, y_pred))
            print("Estimator got Explained Variance %f" % explained_variance_score(y_test, y_pred))

## ==============================================
# # main
## ==============================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='location of test data')
    parser.add_argument("-s", "--svm", help='svm', action='store_true')
    parser.add_argument("-c", "--clustering", help='clustering', action='store_true')
    parser.add_argument("-d", "--decision_tree", help='decision_tree', action='store_true')
    parser.add_argument("-l", "--lasso", help='lasso', action='store_true')
    parser.add_argument("-g", "--gp", help='gaussian_process', action='store_true')
    parser.add_argument("-e", "--estimate_performance", help="Per-benchmark performance estimation", action="store_true")

    args = parser.parse_args()

    normalize_data = True
    label_field = LABEL_FIELD

    if args.estimate_performance:
        # shortcut preprocess() below since estimate_performance() does it itself
        estimate_performance(args.file)

    if args.decision_tree:
        normalize_data = False

    if args.lasso or args.gp:
        label_field = 1         # THROUGHPUT

    if args.file:
        [X, y, num_labels] = preprocess(args.file, normalize_data, label_field)

    get_info(True)

    # CLASSIFICATION

    if args.clustering:
        clustering_classifier(X, y, num_labels)

    if args.svm:
        svm_classifier(X, y)

    if args.decision_tree:
        decision_tree_classifier(X, y, None, None, "tree.pdf")

    # ESTIMATORS

    if args.lasso:
        lasso_estimator(X, y)

    if args.gp:
        gp_estimator(X, y)
