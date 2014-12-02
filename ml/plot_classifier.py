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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

from ml import preprocess
from ml import decision_tree_classifier

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

DTREE_MAX_DEPTHS = [2, 4, 8]
DTREE_MAX_LEAVES = [2, 4, 8]

GRAPH_DIR = './graphs/'
LABEL_FIELD = 0

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

# Run
def run_classifier(X, y, mode):         
    plot_data = {}
    
    if mode == 0:
        params = DTREE_MAX_DEPTHS
        param_name = 'depth'
        axis_name = "Max Tree Depth"
    else:
        params = DTREE_MAX_LEAVES    
        param_name = 'leaves'
        axis_name = "Max Leaf Nodes"

    precision_list = []
    recall_list = []
                    
    for param in params:
        if mode == 0:
            plot_data[param] = decision_tree_classifier(X, y, param, None, GRAPH_DIR + "tree_"+ param_name + "_" + str(param)+".pdf")
        else:
            plot_data[param] = decision_tree_classifier(X, y, None, param, GRAPH_DIR + "tree_"+ param_name + "_" + str(param)+".pdf")
            
        entry = plot_data[param][0].split('\n')[-2].split('    ')

        precision = float(entry[1])
        recall = float(entry[2])
        
        precision_list.append(precision)
        recall_list.append(recall)
        
    pprint.pprint(precision_list)    
        
    plt.plot(params, precision_list, color=OPT_COLORS[2], linewidth=OPT_LINE_WIDTH, marker=OPT_MARKERS[0], markersize=OPT_MARKER_SIZE)
    plt.plot(params, recall_list, color=OPT_COLORS[0], linewidth=OPT_LINE_WIDTH, marker=OPT_MARKERS[0], markersize=OPT_MARKER_SIZE)    
    plt.legend(['Precision', 'Recall'], loc='upper left', fontsize='x-large')

    plt.xlabel(axis_name, fontproperties=LABEL_FP)
    plt.ylabel("Score", fontproperties=LABEL_FP)
    plt.xlim(1.8,8.2)
    plt.ylim(0.0,1.0)
    plt.xticks(fontproperties=TICK_FP)
    plt.yticks(fontproperties=TICK_FP)
        
    plt.savefig(GRAPH_DIR + param_name + '.pdf')
    plt.close()
    
## ==============================================
# # main
## ==============================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='location of test data')

    args = parser.parse_args()
    
    normalize_data = False
    label_field = LABEL_FIELD
        
    if args.file:
        [X, y, num_labels] = preprocess(args.file, normalize_data, label_field)

    run_classifier(X, y, 0)
    run_classifier(X, y, 1)
