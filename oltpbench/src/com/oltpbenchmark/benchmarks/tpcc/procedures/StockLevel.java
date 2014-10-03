package com.oltpbenchmark.benchmarks.tpcc.procedures;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Random;

import org.apache.log4j.Logger;

import com.oltpbenchmark.api.SQLStmt;
import com.oltpbenchmark.benchmarks.tpcc.TPCCConstants;
import com.oltpbenchmark.benchmarks.tpcc.TPCCUtil;
import com.oltpbenchmark.benchmarks.tpcc.TPCCWorker;

public class StockLevel extends TPCCProcedure {

    private static final Logger LOG = Logger.getLogger(StockLevel.class);

	public SQLStmt stockGetDistOrderIdSQL = new SQLStmt("SELECT D_NEXT_O_ID FROM DISTRICT WHERE D_W_ID = ? AND D_ID = ?");

	public SQLStmt stockGetCountStockSQL = new SQLStmt("SELECT COUNT(DISTINCT (S_I_ID)) AS STOCK_COUNT"
			+ " FROM " + TPCCConstants.TABLENAME_ORDERLINE + ", " + TPCCConstants.TABLENAME_STOCK
			+ " WHERE OL_W_ID = ?"
			+ " AND OL_D_ID = ?"
			+ " AND OL_O_ID < ?"
			+ " AND OL_O_ID >= ? - 20"
			+ " AND S_W_ID = ?"
			+ " AND S_I_ID = OL_I_ID" + " AND S_QUANTITY < ?");

	// Stock Level Txn
	private PreparedStatement stockGetDistOrderId = null;
	private PreparedStatement stockGetCountStock = null;

	 public ResultSet run(Connection conn, Random gen,
				int terminalWarehouseID, int numWarehouses,
				int terminalDistrictLowerID, int terminalDistrictUpperID,
				TPCCWorker w) throws SQLException {


		stockGetDistOrderId = this.getPreparedStatement(conn, stockGetDistOrderIdSQL);
		stockGetCountStock= this.getPreparedStatement(conn, stockGetCountStockSQL);

		int threshold = TPCCUtil.randomNumber(10, 20, gen);

		int districtID = TPCCUtil.randomNumber(terminalDistrictLowerID,terminalDistrictUpperID, gen);

		stockLevelTransaction(terminalWarehouseID, districtID, threshold,conn,w);

		return null;
	 }



		private void stockLevelTransaction(int w_id, int d_id, int threshold, Connection conn,TPCCWorker w)
				throws SQLException {
			int o_id = 0;
			// XXX int i_id = 0;
			int stock_count = 0;

			// XXX District dist = new District();
			// XXX OrderLine orln = new OrderLine();
			// XXX Stock stck = new Stock();




			stockGetDistOrderId.setInt(1, w_id);
			stockGetDistOrderId.setInt(2, d_id);
			ResultSet rs = stockGetDistOrderId.executeQuery();

			if (!rs.next())
				throw new RuntimeException("D_W_ID="+ w_id +" D_ID="+ d_id+" not found!");
			o_id = rs.getInt("D_NEXT_O_ID");
			rs.close();
			rs = null;


			stockGetCountStock.setInt(1, w_id);
			stockGetCountStock.setInt(2, d_id);
			stockGetCountStock.setInt(3, o_id);
			stockGetCountStock.setInt(4, o_id);
			stockGetCountStock.setInt(5, w_id);
			stockGetCountStock.setInt(6, threshold);
			rs = stockGetCountStock.executeQuery();

			if (!rs.next())
				throw new RuntimeException("OL_W_ID="+w_id +" OL_D_ID="+d_id+" OL_O_ID="+o_id+" not found!");
			stock_count = rs.getInt("STOCK_COUNT");

			conn.commit();

			rs.close();
			rs = null;

			StringBuilder terminalMessage = new StringBuilder();
			terminalMessage
					.append("\n+-------------------------- STOCK-LEVEL --------------------------+");
			terminalMessage.append("\n Warehouse: ");
			terminalMessage.append(w_id);
			terminalMessage.append("\n District:  ");
			terminalMessage.append(d_id);
			terminalMessage.append("\n\n Stock Level Threshold: ");
			terminalMessage.append(threshold);
			terminalMessage.append("\n Low Stock Count:       ");
			terminalMessage.append(stock_count);
			terminalMessage
					.append("\n+-----------------------------------------------------------------+\n\n");
			if(LOG.isTraceEnabled())LOG.trace(terminalMessage.toString());
		}

}
