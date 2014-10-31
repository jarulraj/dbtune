#!/usr/bin/env python
# Feature generator

from __future__ import print_function
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

import numpy as np
import csv
from options import *
from _rl_accel import fp_str
from lxml import etree

from postgres_driver import get_stats
import numpy as np

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
OLTP_BENCH_DIR = BASE_DIR + "./bench/oltpbench"
OLTP_BENCH = "./oltpbenchmark"
OUTPUT_FILE = "features.csv"

NUM_TRAIN = 100
#BENCHMARKS = ['ycsb', 'tatp', 'twitter', 'auctionmark']
#WEIGHTS = {'ycsb': 6, 'tatp' : 7, 'twitter' : 5, 'auctionmark' : 9}
BENCHMARKS = ['ycsb', 'tatp']
WEIGHTS = {'ycsb': 6, 'tatp' : 7}

# GLOBALS
csv_file = open(OUTPUT_FILE, 'wb')

# Parse SUMMARY
def parse_ob_summary(output, map):
    summary_file_name = output + ".summary"
    summary_file = open(summary_file_name, "r")
    
    for line_cnt, line in enumerate(summary_file):
        if line_cnt == 0:
            map['Timestamp'] = line.strip()   
        if line_cnt == 1:
            map['Database'] = line.strip()
        if line_cnt == 2:
            if 'Benchmark' not in map:
                map['Benchmark'] = line.strip()  
        if line_cnt == 3:
            entry = line.split(',')
            for pair in entry:
                pair = pair.strip(' ').strip("\n").strip('[').strip(']')
                p = pair.split('=')         
                map['Latency_' + p[0]] = p[1]                                       
        if line_cnt == 4:
            map['Throughput'] = line.strip()            
        if line_cnt == 5:
            entry = line.strip().split('=');
            map['Isolation'] = entry[1]    
        if line_cnt == 6:
            entry = line.strip().split('=');
            map['Scalefactor'] = entry[1]    
        if line_cnt == 7:
            entry = line.strip().split('=');
            map['Terminals'] = entry[1]    

# PARSE DB CONF
def parse_db_conf(output, map):
    db_conf_file_name = output + ".db.cnf"
    db_conf_file = open(db_conf_file_name, "r")
    
    for line_cnt, line in enumerate(db_conf_file):
        entry = line.split('=')
        map[entry[0].strip()] = entry[1].strip()                                       
    
# Get weights

def get_weights(benchmark, run):

    if benchmark == 'ycsb':
        ycsb_type = random.randint(1,4)
        ycsb_perturb = random.uniform(0, 3)
        
        if ycsb_type == 1:
            weights = [ 100.0, 0, 0, 0, 0, 0 ]
            run['Benchmark'] = 'ycsb_read_only' 
        elif ycsb_type == 2:
            weights = [ 90.0, 10.0, 0, 0, 0, 0 ]
            run['Benchmark'] = 'ycsb_read_heavy' 
        elif ycsb_type == 3:
            weights = [ 50.0, 50.0, 0, 0, 0, 0 ]
            run['Benchmark'] = 'ycsb_balanced' 
        elif ycsb_type == 4:
            weights = [ 10.0, 90.0, 0, 0, 0, 0 ]            
            run['Benchmark'] = 'ycsb_write_heavy' 

        if ycsb_type != 1:         
            weights[0] = min(100.0, weights[0] + ycsb_perturb)
            weights[1] = max(0.0, weights[1] - ycsb_perturb)
            
        weights_str = ','.join(map(str, weights))    
    else:
        weights = np.random.random(WEIGHTS[benchmark])
        weights /= (weights.sum())
        weights *= 100.0
        weights_str = ','.join(map(str, weights.tolist()))

    pprint.pprint(benchmark);
    pprint.pprint(weights_str);

    return weights_str
       
# Execute OLTP BENCH
def execute_oltpbench():
    LOG.info("Executing OLTP Bench")
    
    def cleanup(prefix):
        files = glob.glob(prefix + "*")
        for f in files:
            os.remove(f)

    log_name = "log.txt"
    log_file = open(log_name, 'w')
    log_file.write('Start :: %s \n' % datetime.datetime.now())
    log_file.flush()

    # Go to config dir
    cwd = os.getcwd()
    os.chdir(OLTP_BENCH_DIR)
 
    for run_itr in range(0, NUM_TRAIN):
 
        # Pick benchmark and generate config file
        benchmark = random.choice(BENCHMARKS)
        pprint.pprint("RUN " + str(run_itr) + " :: " + str(benchmark))       
         
        ob_base_config_file = '../config/' + benchmark + '_config.xml'
        ob_config_file = '../../features/config/test_' + str(run_itr) + '_' + benchmark + '_config.xml' 
    
        tree = etree.parse(ob_base_config_file)
        #tree.find('scalefactor').text = '0.23'
        
        run = {}  

        weights_str = get_weights(benchmark, run);
                             
        tree.find('works').find('work').find('weights').text = weights_str
                 
        out_file = open(ob_config_file, 'w')
        tree.write(out_file)
        out_file.close()

        # Execute oltpbench        
        ob_create = 'true'
        ob_load = 'true'
        ob_execute = 'true'
        ob_window = str(100)
        prefix = 'output'
    
        # ./oltpbenchmark -b ycsb -c ../config/ycsb_config.xml --create=true --load=true --execute=true -s 5 -o
        subprocess.check_call([OLTP_BENCH, '-b', benchmark, '-c', ob_config_file, 
                         '--create', ob_create, '--load', ob_load, '--execute', ob_execute, '-s', ob_window, '-o', prefix],
                              stdout = log_file)
                          
        # Get stats from OLTP Bench
        parse_ob_summary(prefix, run);    

        # Get conf from OLTP Bench
        #parse_db_conf(prefix, run);    

        # Get stats from PG
        get_stats(benchmark, run) 
    
        # Remove empty features
        #run = dict((k, v) for k, v in run.iteritems() if v)
                        
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        wr.writerow(run.values())

        # Cleanup output files
        cleanup(prefix)         
    
    # Write out CSV file        
    log_file.flush()
    log_file.write('End :: %s \n' % datetime.datetime.now())
    log_file.close()
    
    # Go back to orig dir
    os.chdir(cwd)
    


## ==============================================
# # main
## ==============================================
if __name__ == '__main__':
    LOG.info("Feature generator")
    
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument("-y", "--run-ycsb", help='run ycsb', action='store_true')

    args = parser.parse_args()    
             
    execute_oltpbench()

    csv_file.close()
    LOG.info("Done")

    
