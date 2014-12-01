# POSTGRES HOOKS
# Based on postgres-librato/blob/master/publish.py

from __future__ import division
from __future__ import print_function

import psycopg2
import pprint
import time
import json
import sys
import os
import traceback
import subprocess
import logging
import datetime
import re
import glob

import io
from configobj import ConfigObj        

# # CONFIGURATION
BASE_DIR = os.path.dirname(__file__)
PG_CONFIG_DIR = os.path.join(BASE_DIR, "../db/data")
PG_CTL = "pg_ctl"
PG_CONFIG_FILE = PG_CONFIG_DIR + "/postgresql.conf"

def fetch_pg_version(cur):
    cur.execute("SELECT split_part(version(), ' ', 2)")
    res = cur.fetchall()
    val = res[0][0].split("devel")[0]
    val = val.split(".")[1]
    return tuple(map(int, val))

def fetch_index_hits(cur):
    cur.execute("SELECT (sum(idx_blks_hit)) / (1 + sum(idx_blks_hit + idx_blks_read)) AS ratio FROM pg_statio_user_indexes")
    res = cur.fetchall()
    ret = 0.0
    if res[0][0] is not None:
        ret = float(res[0][0])
    return ret

def fetch_cache_hits(cur):
    cur.execute("SELECT sum(heap_blks_hit) / (1 + sum(heap_blks_hit) + sum(heap_blks_read)) AS ratio FROM pg_statio_user_tables")
    res = cur.fetchall()
    ret = 0.0
    if res[0][0] is not None:
        ret = float(res[0][0])
    return ret

def fetch_backend_states(cur, version):
    if version < (9,2):
        cur.execute("""select (case
            when current_query = '<IDLE> in transaction' then 'idle_in_transaction'
            when current_query = '<IDLE>' then 'idle'
            when current_query like 'autovacuum:%' then 'autovacuum'
            else 'active'
            end), count(*) from pg_stat_activity group by 1
            """)
    else:
        cur.execute("select state, count(*) from pg_stat_activity group by 1")
    res = cur.fetchall()
    states = []
    for state, count in res:
        if state is None:
            state = 'null'
        state = state.replace(' ', '_')
        states.append((state, int(count)))
    return states

def fetch_waiting_backends(cur):
    cur.execute("select count(*) from pg_stat_activity where waiting")
    res = cur.fetchall()
    return int(res[0][0])

def fetch_backend_times(cur, version):
    if version < (9,2):
        where = "current_query not like '<IDLE>%' and current_query not like '%pg_stat%' and current_query not like 'autovacuum:%'"
    else:
        where = "state != 'idle' and query not like '%pg_stat%'"
    cur.execute("select extract ('epoch' from GREATEST(now() - query_start, '0')) as runtime from pg_stat_activity where %s order by 1" % where)
    res = cur.fetchall()
    times = [row[0] for row in res]
    if times:
        max_time = max(times)
        mean_time = sum(times) / len(times)
        median_time = times[int(len(times) / 2)]
        return [
            ("max_query_time", max_time),
            ("mean_query_time", mean_time),
            ("median_query_time", median_time),
        ]
    else:
        return []

def fetch_seq_scans(cur):
    cur.execute("SELECT sum(seq_scan), sum(idx_scan) FROM pg_stat_user_tables")
    res = cur.fetchall()
    return [
        ("sequential_scans", res[0][0]),
        ("index_scans", res[0][1])
    ]

# KEY DB STATS
def fetch_db_stats(cur, db, version):
    fields = [
        ("xact_commit", "transactions_committed"),     # Number of transactions in this database that have been committed
        ("xact_rollback", "transactions_rolled_back"), # Number of transactions in this database that have been rolled back
        ("blks_read", "disk_blocks_read"),             # Number of disk blocks read in this database
        ("blks_hit", "disk_blocks_cache_hit"),         # Number of times disk blocks were found already in the buffer cache, so that a read was not necessary (this only includes hits in the PostgreSQL buffer cache, not the operating system's file system cache)
        ("tup_returned", "rows_returned"),             # Number of rows returned by queries in this database
        ("tup_fetched", "rows_fetched"),               # Number of rows fetched by queries in this database
        ("tup_inserted", "rows_inserted"),             # Number of rows inserted by queries in this database
        ("tup_updated", "rows_updated"),               # Number of rows updated by queries in this database
        ("tup_deleted", "rows_deleted"),               # Number of rows deleted by queries in this database
    ]
    if version >= (9,2):
        fields.extend([
            ("temp_bytes", "temp_file_bytes"),         # Total amount of data written to temporary files by queries in this database. All temporary files are counted, regardless of why the temporary file was created, and regardless of the log_temp_files setting.
            ("blk_read_time", "block_read_time"),      # Time spent reading data file blocks by backends in this database, in milliseconds
        ])
    cur.execute("select %s from pg_stat_database where datname = '%s'" % (", ".join(f for f, _ in fields), db))
    res = cur.fetchall()
    row = res[0]
    result = []
    for name, value in zip((name for _, name in fields), row):
        result.append((name, str(long(round(value)))))
    return result

# TODO: Implement
def fetch_index_sizes(cur):
    pass

# TODO: Implement
def fetch_tables_sizes(cur):
    pass

def reset_stats(cur):
    cur.execute("SELECT pg_stat_reset();")
    res = cur.fetchall()
    return res

def get_stats(db, map):
        
    try:
        conn = psycopg2.connect(database=db, user="postgres", password="postgres", host="localhost", port="5432")
    except psycopg2.OperationalError as e:
        print(repr(e))

    cur = conn.cursor()
    try:
        version = fetch_pg_version(cur)
        index_hits = fetch_index_hits(cur)
        cache_hits = fetch_cache_hits(cur)
                
        scans = fetch_seq_scans(cur)
        db_stats = fetch_db_stats(cur, db, version)
        index_sizes = fetch_index_sizes(cur)
        
        reset = reset_stats(cur)
        
        # Normalize
        num_txns = float(db_stats[0][1]) + float(db_stats[1][1]) # commit + rollback
        
        map['PG_Index_Hits'] =  float(index_hits)/num_txns
        map['PG_Cache_Hits'] =  float(cache_hits)/num_txns
        
        for metric, count in scans:
            if count is None:
                map['PG_' +  metric] = 0            
            else:
                map['PG_' +  metric] = float(count)/num_txns
            
        for metric, count in db_stats:
            map['PG_' +  metric] = float(count)/num_txns
        
    except Exception as e:
        print(repr(e))
        traceback.print_exc(file=sys.stdout)    
    
    cur.close()
    conn.close()
    
# Pick val for parameters 
def pick_val(attr):
    parameters = {}
    
    parameters['']
    


    
    
# Mutate PG config and restart    
def mutate_config():

    log_name = "log.txt"
    log_file = open(log_name, 'w')

    try:   
        #pg_ctl -D /home/parallels/git/dbtune/db/data stop 
        #subprocess.check_call([PG_CTL, '-D', PG_CONFIG_DIR, 'stop'], stdout=log_file)              

        # Tweak file
        print(PG_CONFIG_FILE)
        
        config = ConfigObj(PG_CONFIG_FILE)
        print(config.keys())

        print(config.get('shared_buffers'))
        
        config['shared_buffers'] = pick_val('shared_buffers')
        
        print(config.get('shared_buffers'))
    
        config.write()
    
        #pg_ctl -D /home/parallels/git/dbtune/db/data start
        #subprocess.check_call([PG_CTL, '-D', PG_CONFIG_DIR, 'start'], stdout=log_file)              
    
    except subprocess.CalledProcessError, e:
        print(repr(e))
        traceback.print_exc(file=sys.stdout)
        
    sys.exit(0)            

    log_file.close()
    

