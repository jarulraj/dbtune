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

# # CONFIGURATION
BASE_DIR = os.path.dirname(__file__)

BENCHMARK_FIELD = 0
LABEL_FIELD = 0
BENCHMARK_LABEL_FIELD = LABEL_FIELD
THROUGHPUT_LABEL_FIELD = 1
LATENCY_LABEL_FIELD=9
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
    [X, y, num_labels] = preprocess(file, normalize_data, label_field)
    num_samples_list = [5, 10, 100]

    print("===========================================================================")
    print("Using Lasso Regression")
    print("===========================================================================")

    measurements = [make_lasso_measurement(1, 'k', r'$\alpha = 1$'),
                    make_lasso_measurement(1e-3, 'b', r'$\alpha = 1\textrm{e}-3$'),
                    make_lasso_measurement(1e-1, 'g', r'$\alpha = 1\textrm{e}-1$'),
                    make_lasso_measurement(10, 'r', r'$\alpha = 10$')]

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
    matplotlib.rc('text', usetex=True)
    plt.figure(figsize=(5, 5))
    plt.title(title_format.format("Lasso Regression"))
    plt.xlabel("Number of Samples in 2-way CV")
    plt.ylabel(r"$\textrm{R}^2$ Score")
    for measurement in measurements:
        plt.plot(measurement['x'], measurement['y'], measurement['color'], label=measurement['label'])
    plt.legend(loc='upper right')
    plt.savefig("lasso_{0}.pdf".format(file_suffix), format="pdf", dpi=1000)

    print("===========================================================================")
    print("Using Gaussian Processes")
    print("===========================================================================")

    measurements = [make_gaussian_measurement(1e-1, 'k', r'$\theta_0 = 1\textrm{e}-1$'),
                    make_gaussian_measurement(1e-3, 'b', r'$\theta_0 = 1\textrm{e}-3$'),
                    make_gaussian_measurement(1, 'g', r'$\theta_0 = 1$'),
                    make_gaussian_measurement(10, 'r', r'$\theta_0 = 10$')]

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
    matplotlib.rc('text', usetex=True)
    plt.figure(figsize=(5, 5))
    plt.title(title_format.format("Gaussian Process Regression"))
    plt.xlabel("Number of Samples in 2-way CV")
    plt.ylabel(r"$\textrm{R}^2$ Score")
    for measurement in measurements:
        plt.plot(measurement['x'], measurement['y'], measurement['color'], label=measurement['label'])
    plt.legend(loc='upper right')
    plt.savefig("gp_{0}.pdf".format(file_suffix), format="pdf", dpi=1000)

## ==============================================
# # main
## ==============================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='location of test data')

    args = parser.parse_args()

    normalize_data = True
    label_field = LABEL_FIELD

    estimate_performance(args.file, THROUGHPUT_LABEL_FIELD, "Using {0} to Estimate Throughput", "throughput")
    estimate_performance(args.file, LATENCY_LABEL_FIELD, "Using {0} to Estimate Latency", "latency")
