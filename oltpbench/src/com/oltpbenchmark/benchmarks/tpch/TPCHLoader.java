/*******************************************************************************
 * oltpbenchmark.com
 *
 *  Project Info:  http://oltpbenchmark.com
 *  Project Members:  	Carlo Curino <carlo.curino@gmail.com>
 * 				Evan Jones <ej@evanjones.ca>
 * 				DIFALLAH Djellel Eddine <djelleleddine.difallah@unifr.ch>
 * 				Andy Pavlo <pavlo@cs.brown.edu>
 * 				CUDRE-MAUROUX Philippe <philippe.cudre-mauroux@unifr.ch>
 *  				Yang Zhang <yaaang@gmail.com>
 *
 *  This library is free software; you can redistribute it and/or modify it under the terms
 *  of the GNU General Public License as published by the Free Software Foundation;
 *  either version 3.0 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
 *  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the GNU Lesser General Public License for more details.
 ******************************************************************************/

/***
 *   TPC-H implementation
 *
 *   Ben Reilly (bd.reilly@gmail.com)
 *   Ippokratis Pandis (ipandis@us.ibm.com)
 *
 ***/

package com.oltpbenchmark.benchmarks.tpch;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.Date;
import java.util.StringTokenizer;
import java.util.NoSuchElementException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.log4j.Logger;

import com.oltpbenchmark.api.BenchmarkModule;
import com.oltpbenchmark.api.Loader;
import com.oltpbenchmark.util.SimpleSystemPrinter;

import com.oltpbenchmark.benchmarks.tpch.queries.Q1;

public class TPCHLoader extends Loader {
    private static final Logger LOG = Logger.getLogger(TPCHLoader.class);
    private final static int configCommitCount = 10000; // commit every n records
    private static PreparedStatement customerPrepStmt;
    private static PreparedStatement lineitemPrepStmt;
    private static PreparedStatement nationPrepStmt;
    private static PreparedStatement ordersPrepStmt;
    private static PreparedStatement partPrepStmt;
    private static PreparedStatement partsuppPrepStmt;
    private static PreparedStatement regionPrepStmt;
    private static PreparedStatement supplierPrepStmt;
    private static final int numCustomerCols = 8;
    private static final int numLineItemCols = 16;
    private static final int numNationCols = 4;
    private static final int numOrdersCols = 9;
    private static final int numPartCols = 9;
    private static final int numPartSuppCols = 5;
    private static final int numRegionCols = 3;
    private static final int numSupplierCols = 7;

    private static Date now;
    private static long lastTimeMS;
    private static Connection conn;


    public TPCHLoader(BenchmarkModule benchmark, Connection c) {
        super(benchmark, c);
        conn =c;
    }

    private static enum CastTypes { LONG, DOUBLE, STRING, DATE };


    private static final CastTypes[] customerTypes = {
        CastTypes.LONG,   // c_custkey
        CastTypes.STRING, // c_name
        CastTypes.STRING, // c_address
        CastTypes.LONG,   // c_nationkey
        CastTypes.STRING, // c_phone
        CastTypes.DOUBLE, // c_acctbal
        CastTypes.STRING, // c_mktsegment
        CastTypes.STRING  // c_comment
    };

    private static final CastTypes[] lineitemTypes = {
        CastTypes.LONG, // l_orderkey
        CastTypes.LONG, // l_partkey
        CastTypes.LONG, // l_suppkey
        CastTypes.LONG, // l_linenumber
        CastTypes.DOUBLE, // l_quantity
        CastTypes.DOUBLE, // l_extendedprice
        CastTypes.DOUBLE, // l_discount
        CastTypes.DOUBLE, // l_tax
        CastTypes.STRING, // l_returnflag
        CastTypes.STRING, // l_linestatus
        CastTypes.DATE, // l_shipdate
        CastTypes.DATE, // l_commitdate
        CastTypes.DATE, // l_receiptdate
        CastTypes.STRING, // l_shipinstruct
        CastTypes.STRING, // l_shipmode
        CastTypes.STRING  // l_comment
    };

    private static final CastTypes[] nationTypes = {
        CastTypes.LONG,   // n_nationkey
        CastTypes.STRING, // n_name
        CastTypes.LONG,   // n_regionkey
        CastTypes.STRING  // n_comment
    };

    private static final CastTypes[] ordersTypes = {
        CastTypes.LONG,   // o_orderkey
        CastTypes.LONG,   // o_LONG, custkey
        CastTypes.STRING, // o_orderstatus
        CastTypes.DOUBLE, // o_totalprice
        CastTypes.DATE,   // o_orderdate
        CastTypes.STRING, // o_orderpriority
        CastTypes.STRING, // o_clerk
        CastTypes.LONG,   // o_shippriority
        CastTypes.STRING  // o_comment
    };

    private static final CastTypes[] partTypes = {
        CastTypes.LONG,   // p_partkey
        CastTypes.STRING, // p_name
        CastTypes.STRING, // p_mfgr
        CastTypes.STRING, // p_brand
        CastTypes.STRING, // p_type
        CastTypes.LONG,   // p_size
        CastTypes.STRING, // p_container
        CastTypes.DOUBLE, // p_retailprice
        CastTypes.STRING  // p_comment
    };

    private static final CastTypes[] partsuppTypes = {
        CastTypes.LONG,   // ps_partkey
        CastTypes.LONG,   // ps_suppkey
        CastTypes.LONG,   // ps_availqty
        CastTypes.DOUBLE, // ps_supplycost
        CastTypes.STRING  // ps_comment
    };

    private static final CastTypes[] regionTypes = {
        CastTypes.LONG,   // r_regionkey
        CastTypes.STRING, // r_name
        CastTypes.STRING  // r_comment
    };

    private static final CastTypes[] supplierTypes = {
        CastTypes.LONG,   // s_suppkey
        CastTypes.STRING, // s_name
        CastTypes.STRING, // s_address
        CastTypes.LONG,   // s_nationkey
        CastTypes.STRING, // s_phone
        CastTypes.DOUBLE, // s_acctbal
        CastTypes.STRING, // s_comment
    };

    public void load() throws SQLException {
        try {
            customerPrepStmt = conn.prepareStatement("INSERT INTO customer "
                    + "(c_custkey, c_name, c_address, c_nationkey,"
                    + " c_phone, c_acctbal, c_mktsegment, c_comment ) "
                    + "VALUES (?, ?, ?, ?, ?, ?, ?, ?)");

            lineitemPrepStmt = conn.prepareStatement("INSERT INTO lineitem "
                    + "(l_orderkey, l_partkey, l_suppkey, l_linenumber,"
                    + " l_quantity, l_extendedprice, l_discount, l_tax,"
                    + " l_returnflag, l_linestatus, l_shipdate, l_commitdate,"
                    + " l_receiptdate, l_shipinstruct, l_shipmode, l_comment) "
                    + "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)");

            nationPrepStmt = conn.prepareStatement("INSERT INTO nation "
                    + "(n_nationkey, n_name, n_regionkey, n_comment) "
                    + "VALUES (?, ?, ?, ?)");

            ordersPrepStmt = conn.prepareStatement("INSERT INTO orders "
                    + "(o_orderkey, o_custkey, o_orderstatus, o_totalprice,"
                    + " o_orderdate, o_orderpriority, o_clerk, o_shippriority,"
                    + " o_comment) "
                    + "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");

            partPrepStmt = conn.prepareStatement("INSERT INTO part "
                    + "(p_partkey, p_name, p_mfgr, p_brand, p_type,"
                    + " p_size, p_container, p_retailprice, p_comment) "
                    + "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)");

            partsuppPrepStmt = conn.prepareStatement("INSERT INTO partsupp "
                    + "(ps_partkey, ps_suppkey, ps_availqty, ps_supplycost,"
                    + " ps_comment) "
                    + "VALUES (?, ?, ?, ?, ?)");

            regionPrepStmt = conn.prepareStatement("INSERT INTO region "
                    + " (r_regionkey, r_name, r_comment) "
                    + "VALUES (?, ?, ?)");

            supplierPrepStmt = conn.prepareStatement("INSERT INTO supplier "
                    + "(s_suppkey, s_name, s_address, s_nationkey, s_phone,"
                    + " s_acctbal, s_comment) "
                    + "VALUES (?, ?, ?, ?, ?, ?, ?)");

        } catch (SQLException se) {
            LOG.debug(se.getMessage());
            conn.rollback();

        } catch (Exception e) {
            e.printStackTrace();
            conn.rollback();
        } // end try

        loadHelper();
        conn.commit();
    }

    static void truncateTable(String strTable) throws SQLException {
        LOG.debug("Truncating '" + strTable + "' ...");
        try {
            conn.createStatement().execute("DELETE FROM " + strTable);
            conn.commit();
        } catch (SQLException se) {
            LOG.debug(se.getMessage());
            conn.rollback();
        }
    }

    Thread loadCustomers() {
        return new Thread(new CSVLoader("Customer", customerTypes, customerPrepStmt, this));
    }

    Thread loadLineItems() {
        return new Thread(new CSVLoader("LineItem", lineitemTypes, lineitemPrepStmt, this));
    }

    Thread loadNations() {
        return new Thread(new CSVLoader("Nation", nationTypes, nationPrepStmt, this));
    }

    Thread loadOrders() {
        return new Thread(new CSVLoader("Orders", ordersTypes, ordersPrepStmt, this));
    }

    Thread loadParts() {
        return new Thread(new CSVLoader("Part", partTypes, partPrepStmt, this));
    }

    Thread loadPartSupps() {
        return new Thread(new CSVLoader("PartSupp", partsuppTypes, partsuppPrepStmt, this));
    }

    Thread loadRegions() {
        return new Thread(new CSVLoader("Region", regionTypes, regionPrepStmt, this));
    }

    Thread loadSuppliers() {
        return new Thread(new CSVLoader("Supplier", supplierTypes, supplierPrepStmt, this));
    }

    protected long totalRows = 0;

    protected long loadHelper() {
        Thread loaders[] = new Thread[8];
        loaders[0] = loadCustomers();
        loaders[1] = loadLineItems();
        loaders[2] = loadNations();
        loaders[3] = loadOrders();
        loaders[4] = loadParts();
        loaders[5] = loadPartSupps();
        loaders[6] = loadRegions();
        loaders[7] = loadSuppliers();

        for (int i = 0; i < 8; ++i)
            if (loaders[i] != null)
                loaders[i].start();

        for (int i = 0; i < 8; ++i) {
            try {
                if (loaders[i] != null)
                    loaders[i].join();
            } catch(InterruptedException e) {
                LOG.error(e.getMessage());
            }
        }

        return this.totalRows;
    }

    private class CSVLoader implements Runnable {
        String tableName;
        PreparedStatement prepStmt;
        CastTypes[] types;
        final TPCHLoader parent;

        private Connection conn;

        CSVLoader(String tableName, CastTypes[] types
                  , PreparedStatement prepStmt, TPCHLoader parent)
        {
            this.tableName = tableName;
            this.prepStmt = prepStmt;
            this.types = types;
            this.parent = parent;
        }

        public void run() {
            BufferedReader br = null;
            int recordsRead = 0;
            long lastTimeMS = new java.util.Date().getTime();

            try {
                truncateTable(this.tableName.toLowerCase());
            } catch (SQLException e) {
                LOG.error("Failed to truncate table \""
                        + this.tableName.toLowerCase()
                        + "\".");
            }

            try {
                this.conn = DriverManager.getConnection(workConf.getDBConnection(),
                        workConf.getDBUsername(),
                        workConf.getDBPassword());
                this.conn.setAutoCommit(false);

                try {
                    now = new java.util.Date();
                    LOG.debug("\nStart " + tableName + " load @ " + now + "...");

                    File file = new File(workConf.getDataDir()
                                         , tableName.toLowerCase() + ".csv");
                    br = new BufferedReader(new FileReader(file));
                    String line;
                    // The following pattern parses the lines by commas, except for
                    // ones surrounded by double-quotes. Further, strings that are
                    // double-quoted have the quotes dropped (we don't need them).
                    Pattern csvPattern = Pattern.compile("\\s*(\"[^\"]*\"|[^,]*)\\s*,?");
                    Matcher matcher = null;
                    while ((line = br.readLine()) != null) {
                        matcher = csvPattern.matcher(line);
                        try {
                            for (int i = 0; i < types.length; ++i) {
                                matcher.find();
                                String field = matcher.group(1);

                                // Remove quotes that may surround a field.
                                if (field.charAt(0) == '\"') {
                                    field = field.substring(1, field.length() - 1);
                                }

                                switch(types[i]) {
                                    case DOUBLE:
                                        prepStmt.setDouble(i+1, Double.parseDouble(field));
                                        break;
                                    case LONG:
                                        prepStmt.setLong(i+1, Long.parseLong(field));
                                        break;
                                    case STRING:
                                        prepStmt.setString(i+1, field);
                                        break;
                                    case DATE:
                                        // Four possible formats for date
                                        // yyyy-mm-dd
                                        Pattern isoFmt = Pattern.compile("^\\s*(\\d{4})-(\\d{2})-(\\d{2})\\s*$");
                                        Matcher isoMatcher = isoFmt.matcher(field);
                                        // yyyymmdd
                                        Pattern nondelimFmt = Pattern.compile("^\\s*(\\d{4})(\\d{2})(\\d{2})\\s*$");
                                        Matcher nondelimMatcher = nondelimFmt.matcher(field);
                                        // mm/dd/yyyy
                                        Pattern usaFmt = Pattern.compile("^\\s*(\\d{2})/(\\d{2})/(\\d{4})\\s*$");
                                        Matcher usaMatcher = usaFmt.matcher(field);
                                        // dd.mm.yyyy
                                        Pattern eurFmt = Pattern.compile("^\\s*(\\d{2})\\.(\\d{2})\\.(\\d{4})\\s*$");
                                        Matcher eurMatcher = eurFmt.matcher(field);

                                        java.sql.Date fieldAsDate = null;
                                        if (isoMatcher.find()) {
                                            fieldAsDate = new java.sql.Date(
                                                    Integer.parseInt(isoMatcher.group(1)) - 1900,
                                                    Integer.parseInt(isoMatcher.group(2)),
                                                    Integer.parseInt(isoMatcher.group(3)));
                                        }
                                        else if (nondelimMatcher.find()) {
                                            fieldAsDate = new java.sql.Date(
                                                    Integer.parseInt(nondelimMatcher.group(1)) - 1900,
                                                    Integer.parseInt(nondelimMatcher.group(2)),
                                                    Integer.parseInt(nondelimMatcher.group(3)));
                                        }
                                        else if (usaMatcher.find()) {
                                            fieldAsDate = new java.sql.Date(
                                                    Integer.parseInt(usaMatcher.group(3)) - 1900,
                                                    Integer.parseInt(usaMatcher.group(1)),
                                                    Integer.parseInt(usaMatcher.group(2)));
                                        }
                                        else if (eurMatcher.find()) {
                                            fieldAsDate = new java.sql.Date(
                                                    Integer.parseInt(eurMatcher.group(3)) - 1900,
                                                    Integer.parseInt(eurMatcher.group(2)),
                                                    Integer.parseInt(eurMatcher.group(1)));
                                        }
                                        else {
                                            throw new RuntimeException("Unrecognized date \""
                                                + field + "\" in CSV file: "
                                                + file.getAbsolutePath());
                                        }
                                        prepStmt.setDate(i+1, fieldAsDate, null);
                                        break;
                                    default:
                                        throw new RuntimeException("Unrecognized type for prepared statement");
                                }
                            }
                        } catch(IllegalStateException e) {
                            // This happens if there wasn't a match against the regex.
                            LOG.error("Invalid CSV file: " + file.getAbsolutePath());
                        }

                        prepStmt.addBatch();
                        ++recordsRead;

                        if ((recordsRead % configCommitCount) == 0) {
                            long currTime = new java.util.Date().getTime();
                            String elapsedStr = "  Elapsed Time(ms): "
                                + ((currTime - lastTimeMS) / 1000.000)
                                + "                    ";
                            LOG.debug(elapsedStr.substring(0,30)
                                    + "  Writing record " + recordsRead);
                            lastTimeMS = currTime;
                            prepStmt.executeBatch();
                            prepStmt.clearBatch();
                            conn.commit();
                        }
                    }

                    long currTime = new java.util.Date().getTime();
                    String elapsedStr = "  Elapsed Time(ms): "
                        + ((currTime - lastTimeMS) / 1000.000)
                        + "                    ";
                    LOG.debug(elapsedStr.substring(0,30)
                            + "  Writing record " + recordsRead);
                    lastTimeMS = currTime;
                    prepStmt.executeBatch();
                    conn.commit();
                    now = new java.util.Date();
                    LOG.debug("End " + tableName + " Load @ " + now);

                } catch (SQLException se) {
                    LOG.debug(se.getMessage());
                    se = se.getNextException();
                    LOG.debug(se.getMessage());
                    conn.rollback();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }  catch (Exception e) {
                    e.printStackTrace();
                    conn.rollback();
                } finally {
                    if (br != null){
                        try {
                            br.close();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                }

                synchronized(parent) {
                    parent.totalRows += recordsRead;
                }
                this.conn.close();
            } catch(SQLException e) {
                LOG.debug(e.getMessage());
            }
        }

    };
}
