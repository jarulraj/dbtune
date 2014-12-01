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
import options
from lxml import etree
import numpy as np
from collections import OrderedDict

from cpuinfo import cpuinfo
import psutil

from postgres_driver import get_stats
from postgres_driver import mutate_config

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

NUM_RUNS = 10
#BENCHMARKS = ['ycsb', 'tatp', 'twitter', 'auctionmark', 'epinions', 'tpcc', 'seats', 'wikipedia']
#WEIGHTS = {'ycsb': 6, 'tatp' : 7, 'twitter' : 5, 'auctionmark' : 9, 'epinions' : 9, 'tpcc' : 5, 'seats' : 6, 'wikipedia' : 5}
BENCHMARKS = ['ycsb', 'tatp', 'twitter', 'auctionmark', 'epinions', 'seats', 'wikipedia']
WEIGHTS = {'ycsb': 6, 'tatp' : 7, 'twitter' : 5, 'auctionmark' : 9, 'epinions' : 9, 'seats' : 6, 'wikipedia' : 5}

# GLOBALS
csv_file = None

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
            if 'AA_Benchmark' not in map:
                map['AA_Benchmark'] = line.strip()
        if line_cnt == 3:
            entry = line.split(',')
            for pair in entry:
                pair = pair.strip(' ').strip("\n").strip('[').strip(']')
                p = pair.split('=')
                map['Latency_' + p[0]] = p[1]
        if line_cnt == 4:
            map['AA_Throughput'] = line.strip()
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

# SYS CONF
def get_sys_conf(map):

    info = cpuinfo.get_cpu_info()
    #pprint.pprint(info)

    map['SYS_hz_actual'] = info['hz_actual']
    map['SYS_raw_arch_string'] = info['raw_arch_string']
    map['SYS_l2_cache_size'] = info['l2_cache_size']
    map['SYS_brand'] = info['brand']
    map['SYS_cpu_count'] = info['count']

    meminfo = psutil.virtual_memory()
    #pprint.pprint(meminfo)
    map['SYS_total_mem'] = float(meminfo[0])
    map['SYS_percent_free'] = meminfo[2]

# Get weights
def get_weights(benchmark, run):

    if benchmark == 'ycsb':
        ycsb_type = random.randint(1,4)
        ycsb_perturb = random.uniform(0, 5)

        if ycsb_type == 1:
            weights = [ 0, 100, 0, 0, 0, 0 ]
            run['AA_Benchmark'] = 'ycsb_read_only'
        elif ycsb_type == 2:
            weights = [ 80.0, 20.0, 0, 0, 0, 0 ]
            run['AA_Benchmark'] = 'ycsb_read_heavy'
        elif ycsb_type == 3:
            weights = [ 50.0, 50.0, 0, 0, 0, 0 ]
            run['AA_Benchmark'] = 'ycsb_balanced'
        elif ycsb_type == 4:
            weights = [ 20.0, 80.0, 0, 0, 0, 0 ]
            run['AA_Benchmark'] = 'ycsb_write_heavy'

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
def execute_oltpbench(num_runs, mutate, long_run):
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

    for run_itr in range(0, num_runs):

        # Mutate config
        if mutate:
            mutate_config()

        # Pick benchmark and generate config file
        benchmark = random.choice(BENCHMARKS)
        pprint.pprint("RUN " + str(run_itr) + " :: " + str(benchmark))

        ob_base_config_file = '../config/' + benchmark + '_config.xml'
        ob_config_file = '../../generator/config/test_' + str(run_itr) + '_' + benchmark + '_config.xml'

        tree = etree.parse(ob_base_config_file)
        #tree.find('scalefactor').text = '0.23'

        run = {}

        weights_str = get_weights(benchmark, run);

        tree.find('works').find('work').find('weights').text = weights_str

        if long_run:
            tree.find('works').find('work').find('time').text = "60"
        else:
            tree.find('works').find('work').find('time').text = "3"

        out_file = open(ob_config_file, 'w')
        tree.write(out_file)
        out_file.close()

        # Execute oltpbench
        ob_create = 'true'
        ob_load = 'true'
        ob_execute = 'true'
        ob_window = str(100)
        prefix = 'output'

        try:
            if benchmark == "seats":
                # ./oltpbenchmark -b seats -c ../config/seats_config.xml --execute=true -s 5 -o
                subprocess.check_call([OLTP_BENCH,
                                       '-b', benchmark,
                                       '-c', ob_config_file,
                                       '--execute', ob_execute,
                                       '-s', ob_window,
                                       '-o', prefix],
                                      stdout = log_file)
            else:
                # ./oltpbenchmark -b ycsb -c ../config/ycsb_config.xml --create=true --load=true --execute=true -s 5 -o
                subprocess.check_call([OLTP_BENCH,
                                       '-b', benchmark,
                                       '-c', ob_config_file,
                                       '--create', ob_create,
                                       '--load', ob_load,
                                       '--execute', ob_execute,
                                       '-s', ob_window,
                                       '-o', prefix],
                                      stdout = log_file)
        except subprocess.CalledProcessError, e:
            continue

        # Get stats from OLTP Bench
        parse_ob_summary(prefix, run);

        # Get conf from OLTP Bench
        parse_db_conf(prefix, run);

        # Get system get
        #conf_sys_conf(run);

        # Get stats from PG
        get_stats(benchmark, run)

        # Remove empty features
        #run = dict((k, v) for k, v in run.iteritems() if v)

        run = OrderedDict(sorted(run.items(), key=lambda t: t[0]))
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

        if run_itr == 0:
            wr.writerow(run.keys())
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
    parser.add_argument('-r', '--num_runs', type=str, help='num runs')
    parser.add_argument('-m', '--mutate', help='mutate config', action='store_true')
    parser.add_argument('-l', '--long', help='long running benchmark', action='store_true')

    args = parser.parse_args()

    num_runs = NUM_RUNS
    mutate = False
    long_run = False
    output_file = "data.csv"

    if args.num_runs:
        num_runs = int(args.num_runs)

    if args.mutate:
        mutate = True

    if args.long:
        long_run = True
        output_file = "data_long.csv"

    csv_file = open(output_file, 'wb')

    execute_oltpbench(num_runs, mutate, long_run)

    csv_file.close()
    LOG.info("Done")
