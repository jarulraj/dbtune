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

import numpy as np
import csv
from options import *
from _rl_accel import fp_str

from postgres_driver import get_stats

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
    
# Execute OLTP BENCH
def execute_oltpbench(csv_file):
    LOG.info("Executing OLTP Bench")

    def cleanup(prefix):
        files = glob.glob(prefix + "*")
        for f in files:
            os.remove(f)

    log_name = "log.txt"
    log_file = open(log_name, 'w')
    log_file.write('Start :: %s \n' % datetime.datetime.now())
    log_file.flush()

    cwd = os.getcwd()
    os.chdir(OLTP_BENCH_DIR)

    benchmark = 'ycsb'
    ob_config_file = '../config/ycsb_config.xml'
    ob_create = 'true'
    ob_load = 'true'
    ob_execute = 'true'
    ob_window = str(100)
    prefix = 'output'

    # EXECUTE OLTPBENCHMARK
    # ./oltpbenchmark -b ycsb -c ../config/ycsb_config.xml --create=true --load=true --execute=true -s 5 -o
    subprocess.check_call([OLTP_BENCH, '-b', benchmark, '-c', ob_config_file, 
                     '--create', ob_create, '--load', ob_load, '--execute', ob_execute, '-s', ob_window, '-o', prefix],
                          stdout = log_file)

    run = {}        
    parse_ob_summary(prefix, run);    
    #parse_db_conf(prefix, run);    

    get_stats("ycsb", run) 

    # remove keys with no values
    run = dict((k, v) for k, v in run.iteritems() if v)

    cleanup(prefix)
     
    pprint.pprint(run) 
            
    wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    wr.writerow(run.values())
            
    log_file.flush()
    log_file.write('End :: %s \n' % datetime.datetime.now())
    log_file.close()
    
    os.chdir(cwd)
    


## ==============================================
# # main
## ==============================================
if __name__ == '__main__':
    LOG.info("Feature generator")
    
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument("-y", "--run-ycsb", help='run ycsb', action='store_true')

    args = parser.parse_args()    

    csv_file = open("features.csv", 'wb')
             
    execute_oltpbench(csv_file)

    csv_file.close()
    LOG.info("Done")

    
