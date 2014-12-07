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
from matplotlib.font_manager import FontProperties

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

# # CONFIGURATION
BASE_DIR = os.path.dirname(__file__)
NUM_FOLDS = 5

GRAPH_DIR = './graphs/'

OPT_LABEL_WEIGHT = 'bold'
OPT_LINE_WIDTH = 6.0
OPT_MARKER_SIZE = 10.0
OPT_COLORS = ( '#F58A87', '#80CA86', '#9EC9E9', "#F15854", "#66A26B", "#5DA5DA")
OPT_MARKERS = (['o', 's', 'v', ">", "h", "v", "^", "x", "d", "<", "|", "8", "|", "_"])
OPT_FONT_NAME = 'Arial'
OPT_GRAPH_HEIGHT = 300
OPT_GRAPH_WIDTH = 400

# SET FONT

LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 18

LABEL_FP = FontProperties(family=OPT_FONT_NAME, style='normal', size=LABEL_FONT_SIZE, weight='bold')
TICK_FP = FontProperties(family=OPT_FONT_NAME, style='normal', size=TICK_FONT_SIZE)

BENCHMARK_LABEL_FIELD = "AA_Benchmark"
THROUGHPUT_LABEL_FIELD = "AA_Throughput"
LATENCY_AVG_LABEL_FIELD = "Latency_avg"
NUM_FOLDS = 5

index_to_feature_map = []
index_to_benchmark_map = []

# Get info
def get_info(feature_info):
    # FEATURE LIST
    if feature_info:
        print("Index to Feature Map in X:")
        for index, feature in enumerate(index_to_feature_map):
            print("  {0:>3} : {1}".format(index, feature))

    # BENCHMARK LIST
    print("Index to Benchmake Map")
    for index, benchmark in enumerate(index_to_benchmark_map):
            print("  {0:>3} : {1}".format(index, benchmark))

# Preprocess feature data
def preprocess(filename, normalize_data, label_field, features_to_discard):
    X = np.array([])
    y = np.array([])
    num_features = None
    feature_to_index_map = {}
    global index_to_feature_map
    global index_to_benchmark_map

    with open(filename, "rb") as inputfile:
        text_feature_map = {}
        next_text_feature_number = {}
        rawdata = csv.reader(inputfile)

        # Convert data
        for i, rawdata_i in enumerate(rawdata):
            length = len(rawdata_i)

            if i == 0:
                num_features = length
                f_list = rawdata_i

                if num_features < 2:
                    sys.exit("Need at least one feature and exactly one label!")

                print("Detected " + str(num_features) + " features")
                X = X.reshape(0, num_features)

                for index, item in enumerate(f_list):
                    feature_to_index_map[item] = index
                    index_to_feature_map.append(item)

                continue

            # Check row length
            if length != (num_features):
                sys.exit("Row " + str(i + 1) + " has " + str(length) + " elements! Expected " +
                         str(num_features) + " elements")

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

            # append converted row to X
            X = np.vstack((X, converted_row))

    # extract labels
    y = X[:, feature_to_index_map[label_field]]
    print("Label field is \"{0}\", at index {1}".format(label_field, feature_to_index_map[label_field]))

    # Discard features and labels from X
    print("Discarding {0} from features".format(features_to_discard))
    features_to_discard.append(label_field)
    indices_to_keep = filter(lambda index: not index_to_feature_map[index] in features_to_discard,
                             range(X.shape[1]))
    X = X[:, indices_to_keep]

    # Update index_to_feature_map
    new_index_to_feature_map = []
    for index in indices_to_keep:
        new_index_to_feature_map.append(index_to_feature_map[index])
    index_to_feature_map = new_index_to_feature_map

    # Normalize features
    if normalize_data:
        X = StandardScaler().fit_transform(X)

    # Count number of labels
    y_counter = Counter()
    for label in y:
        y_counter[label] += 1
    num_labels = len(y_counter.keys())

    # BENCHMARK LIST
    benchmark_list = text_feature_map[feature_to_index_map[BENCHMARK_LABEL_FIELD]]
    benchmark_list = sorted(benchmark_list.items(), key=lambda x:x[1])
    for _ in benchmark_list:
        index_to_benchmark_map.append("")
    for benchmark, index in benchmark_list:
        index_to_benchmark_map[index] = benchmark

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

def make_lasso_measurement(alpha, color, label):
    return {'alpha': alpha,
            'x': [],
            'y': [],
            'color': color,
            'label': label}

def make_gaussian_measurement(theta0, color, label):
    return {'theta0': theta0,
            'x': [],
            'y': [],
            'color': color,
            'label': label}

def estimate_performance(file, label_field, title_format, file_suffix):
    [X, y, num_labels] = preprocess(file, normalize_data, label_field, label_field == LATENCY_AVG_LABEL_FIELD)
    num_samples_list = [10, 50, 100, 250, 500, 1000]

    print("===========================================================================")
    print("Using Lasso Regression")
    print("===========================================================================")

    measurements = [make_lasso_measurement(1, 'k', r'$\alpha = 1$'),
                    #make_lasso_measurement(1e-3, 'b', r'$\alpha = 0.001$'),
                    make_lasso_measurement(1e-1, 'g', r'$\alpha = 0.1$')]


    ############
    # Get Data #
    ############
    for measurement in measurements:
        for num_samples in num_samples_list:
            print("Fitting for alpha = {0} across {1} samples".format(measurement['alpha'], num_samples))

            clf = linear_model.Lasso(alpha = measurement['alpha'])
            [X_train, y_train, X_test, y_test] = split_data(X[:num_samples, :], y[:num_samples], 2)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            measurement['x'].append(num_samples)
            measurement['y'].append(r2_score(y_test, y_pred))

    #############
    # Plot Data #
    #############
    #matplotlib.rc('text', usetex=True)
    plt.figure()
    #plt.title(title_format.format("Lasso Regression"))
    plt.xlabel("Number of Samples", fontproperties=LABEL_FP)
    plt.ylabel("R-Squared Score", fontproperties=LABEL_FP)
    for idx, measurement in enumerate(measurements):
        plt.plot(measurement['x'], measurement['y'], label=measurement['label'], color=OPT_COLORS[idx], linewidth=OPT_LINE_WIDTH, marker=OPT_MARKERS[0], markersize=OPT_MARKER_SIZE)
    plt.legend(loc='lower right', fontsize='x-large')
    plt.xticks(fontproperties=TICK_FP)
    plt.yticks(fontproperties=TICK_FP)
    #plt.xlim(1.8,8.2)
    #plt.ylim(0.0,1.0)
    plt.savefig(GRAPH_DIR + "lasso_{0}.pdf".format(file_suffix), format="pdf", dpi=1000)
    plt.close()

    print("===========================================================================")
    print("Using Gaussian Processes")
    print("===========================================================================")

    measurements = [make_gaussian_measurement(1e-1, 'k', r'$\theta_0 = 0.1$'),
                    #make_gaussian_measurement(1e-3, 'b', r'$\theta_0 = 0.001$'),
                    make_gaussian_measurement(1, 'g', r'$\theta_0 = 1$')]

    ############
    # Get Data #
    ############
    for measurement in measurements:
        for num_samples in num_samples_list:
            print("Fitting for theta0 = {0} across {1} samples".format(measurement['theta0'], num_samples))

            clf = gaussian_process.GaussianProcess(theta0 = measurement['theta0'])
            [X_train, y_train, X_test, y_test] = split_data(X[:num_samples, :], y[:num_samples], 2)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            measurement['x'].append(num_samples)
            measurement['y'].append(r2_score(y_test, y_pred))

    #############
    # Plot Data #
    #############
    #matplotlib.rc('text', usetex=True)
    plt.figure()
    #plt.title(title_format.format("Gaussian Process Regression"))
    plt.xlabel("Number of Samples", fontproperties=LABEL_FP)
    plt.ylabel("R-Squared Score", fontproperties=LABEL_FP)
    for idx, measurement in enumerate(measurements):
        plt.plot(measurement['x'], measurement['y'], label=measurement['label'], color=OPT_COLORS[idx], linewidth=OPT_LINE_WIDTH, marker=OPT_MARKERS[0], markersize=OPT_MARKER_SIZE)
    plt.legend(loc='lower right', fontsize='x-large')
    #plt.xlim(1.8,8.2)
    #plt.ylim(0.0,1.0)
    plt.xticks(fontproperties=TICK_FP)
    plt.yticks(fontproperties=TICK_FP)
    plt.savefig(GRAPH_DIR + "gp_{0}.pdf".format(file_suffix), format="pdf", dpi=1000)
    plt.close()

def per_benchmark_gp(file, label_field, title_format, hist_x_label, hist_range, num_bins, hist_ylim, file_suffix, features_to_discard):
    clf = gaussian_process.GaussianProcess(theta0=1e-2, corr='absolute_exponential', regr='constant')

    [_, y_benchmark_all, num_benchmarks] =  preprocess(file, normalize_data, BENCHMARK_LABEL_FIELD, features_to_discard)
    [X_all, y_all, _] = preprocess(file, normalize_data, label_field, features_to_discard)
    num_samples_list = range(500, 1100, 100)

    r2_scores_median = []
    r2_scores_stddev = []
    explained_variances_median = []
    explained_variances_stddev = []
    for num_samples in num_samples_list:
        print("##################################")
        print("Running for {0} samples".format(num_samples))
        print("##################################")
        r2_scores = []
        explained_variances = []
        X = X_all[:num_samples, :]
        y = y_all[:num_samples]
        y_benchmark = y_benchmark_all[:num_samples]
        for benchmark_number in range(num_benchmarks):
            sample_filter = y_benchmark == benchmark_number
            X_filtered = X[sample_filter, :]
            y_filtered = y[sample_filter]

            print("-----------------------")
            print("Estimating for benchmark {0} with {1} samples".format(index_to_benchmark_map[benchmark_number],
                                                                         X_filtered.shape[0]))
            print("-----------------------")

            [X_train, y_train, X_test, y_test] = split_data(X_filtered, y_filtered, 2)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            np.set_printoptions(suppress=True)
            #print(y_test[:20])
            #print(y_pred[:20])

            print("Estimator got R2 Score %f" % r2_score(y_test, y_pred))
            print("Estimator got Explained Variance %f" % explained_variance_score(y_test, y_pred))

            r2_scores.append(r2_score(y_test, y_pred))
            explained_variances.append(explained_variance_score(y_test, y_pred))

        row_format = "{:>15} " * num_benchmarks
        print(row_format.format(*index_to_benchmark_map))
        print(row_format.format(*r2_scores))

        r2_scores_median.append(np.median(r2_scores))
        r2_scores_stddev.append(np.std(r2_scores))
        explained_variances_median.append(np.median(explained_variances))
        explained_variances_stddev.append(np.std(explained_variances))

    for benchmark_number in range(num_benchmarks):
        sample_filter = y_benchmark == benchmark_number
        X_filtered = X[sample_filter, :]
        y_filtered = y[sample_filter]

        print("##################################")
        print("Plotting histogram for benchmark {0} with {1} samples".format(
            index_to_benchmark_map[benchmark_number],
            X_filtered.shape[0]))
        print("##################################")

        [X_train, y_train, X_test, y_test] = split_data(X_filtered, y_filtered, 2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        np.set_printoptions(suppress=True)
        #print(y_test[:20])
        #print(y_pred[:20])

        plt.figure()
        plt.xlabel(hist_x_label, fontproperties=LABEL_FP)
        plt.ylabel("Number of Samples", fontproperties=LABEL_FP)
        plt.hist(y_test, range=hist_range, bins=num_bins, color=OPT_COLORS[2])
        plt.ylim(hist_ylim)
        plt.savefig(GRAPH_DIR + "{0}_test_hist_{1}.pdf".format(index_to_benchmark_map[benchmark_number],
                                                               file_suffix),
                    format="pdf",
                    dpi=1000)
        plt.close()

        plt.figure()
        plt.xlabel(hist_x_label, fontproperties=LABEL_FP)
        plt.ylabel("Number of Samples", fontproperties=LABEL_FP)
        plt.hist(y_pred, range=hist_range, bins=num_bins, color=OPT_COLORS[2])
        plt.ylim(hist_ylim)
        plt.savefig(GRAPH_DIR + "{0}_pred_hist_{1}.pdf".format(index_to_benchmark_map[benchmark_number],
                                                               file_suffix),
                    format="pdf",
                    dpi=1000)
        plt.close()

    plt.figure()
    plt.xlabel("Number of Samples", fontproperties=LABEL_FP)
    plt.ylabel("R-Squared Score", fontproperties=LABEL_FP)
    #plt.errorbar(num_samples_list, r2_scores_median, yerr=r2_scores_stddev, color=OPT_COLORS[0], linewidth=OPT_LINE_WIDTH, marker=OPT_MARKERS[0], markersize=OPT_MARKER_SIZE)
    plt.plot(num_samples_list, r2_scores_median, color=OPT_COLORS[0], linewidth=OPT_LINE_WIDTH, marker=OPT_MARKERS[0], markersize=OPT_MARKER_SIZE)
    plt.fill_between(num_samples_list, np.array(r2_scores_median) - np.array(r2_scores_stddev), np.array(r2_scores_median) + np.array(r2_scores_stddev), color='#ccffcc', linewidth=OPT_LINE_WIDTH)
    #plt.xlim(0, 1100)
    #plt.ylim(-1.5, 1.5)
    plt.xticks(fontproperties=TICK_FP)
    plt.yticks(fontproperties=TICK_FP)
    plt.savefig(GRAPH_DIR + "gp_per_benchmark_r2_scores_{0}.pdf".format(file_suffix), format="pdf", dpi=1000)
    plt.close()

    plt.figure()
    plt.xlabel("Number of Samples", fontproperties=LABEL_FP)
    plt.ylabel("Explained Variance", fontproperties=LABEL_FP)
    plt.errorbar(num_samples_list, explained_variances_median, yerr=explained_variances_stddev, color=OPT_COLORS[0], linewidth=OPT_LINE_WIDTH, marker=OPT_MARKERS[0], markersize=OPT_MARKER_SIZE)
    #plt.xlim(0, 1100)
    #plt.ylim(-1.5, 1.5)
    plt.xticks(fontproperties=TICK_FP)
    plt.yticks(fontproperties=TICK_FP)
    plt.savefig(GRAPH_DIR + "gp_per_benchmark_explained_variances_{0}.pdf".format(file_suffix), format="pdf", dpi=1000)
    plt.close()

## ==============================================
# # main
## ==============================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='location of test data')
    parser.add_argument('-m', '--mutate', help='mutate config', action='store_true')

    args = parser.parse_args()

    normalize_data = True
    suffix = ""
    if args.mutate:
        suffix = "_mutate"

    features_to_discard = ["AA_Throughput",
                           "Latency_25th",
                           "Latency_75th",
                           "Latency_90th",
                           "Latency_95th",
                           "Latency_99th",
                           "Latency_avg",
                           "Latency_max",
                           "Latency_median",
                           "Latency_min",
                           "Scalefactor",
                           "Isolation",
                           "Terminals"]

    per_benchmark_gp(args.file, THROUGHPUT_LABEL_FIELD, "Using {0} to Estimate Throughput", "Throughput (transactions/second)", (-2000, 10000), 12, (0, 175), "throughput" + suffix, features_to_discard)
    per_benchmark_gp(args.file, LATENCY_AVG_LABEL_FIELD, "Using {0} to Estimate Latency", "Latency (milliseconds)", (-2, 10), 12, (0, 175), "latency" + suffix, features_to_discard)
