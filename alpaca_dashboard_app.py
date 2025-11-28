import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
load_dotenv()


# ===============================
# 🔧 ENV + DATABASE CONNECT
# ===============================

DB_URL = os.getenv("DB_URL")  # MUST be set in Render environment

engine = None
DB_CONNECTED = False

if DB_URL:
    try:
        engine = create_engine(DB_URL)
        conn_test = engine.connect()
        conn_test.close()
        DB_CONNECTED = True
    except Exception as e:
        DB_CONNECTED = False
else:
    DB_CONNECTED = False


# ===============================
# 📥 DATABASE READ FUNCTIONS
# ===============================

def load_trades():
    if not DB_CONNECTED:
        return pd.DataFrame()
    try:
        return pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC", engine)
    except Exception as e:
        st.error(f"❌ Error reading trades table: {e}")
        return pd.DataFrame()


def load_equity():
    if not DB_CONNECTED:
        return pd.DataFrame()
    try:
        return pd.read_sql("SELECT * FROM equity ORDER BY timestamp ASC", engine)
    except Exception as e:
        st.error(f"❌ Error reading equity table: {e}")
        return pd.DataFrame()


# ===============================
# 🎨 UI DASHBOARD
# ===============================

def main():
    st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
    st.title("🚀 AI Trading Live Dashboard")

    st.write("Connected to: **Alpaca + PostgreSQL + Railway Trading Bot**")

    # --- STATUS BOX ---
    if DB_CONNECTED:
        st.success("🟢 Database Connected Successfully")
    else:
        st.error("🔴 DATABASE NOT CONNECTED — DB_URL missing or invalid.\n\n"
                 "➡ Add `DB_URL` in Render -> Environment Variables.")

    trades = load_trades()
    equity = load_equity()

    # =======================
    # Metrics Section
    # =======================
    st.subheader("📊 Performance Summary")

    if not equity.empty:
        equity["timestamp"] = pd.to_datetime(equity["timestamp"])
        latest_equity = equity["equity"].iloc[-1]
        starting_equity = equity["equity"].iloc[0]
        total_return = ((latest_equity / starting_equity) - 1) * 100
    else:
        latest_equity = 0
        total_return = 0

    closed_trades = trades[trades["status"] == "closed"] if not trades.empty else pd.DataFrame()
    wins = len(closed_trades[closed_trades["pnl"] > 0]) if not closed_trades.empty else 0
    total_closed = len(closed_trades)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Current Equity", f"${latest_equity:,.2f}")
    col2.metric("📈 Total Return", f"{total_return:.2f}%")
    col3.metric("📉 Trades Count", len(trades))
    col4.metric("🏆 Win Rate", f"{(wins / total_closed * 100):.2f}%" if total_closed else "N/A")

    # =======================
    # Equity Chart
    # =======================
    st.subheader("📈 Equity Curve")

    if not equity.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(equity["timestamp"], equity["equity"], linewidth=2)

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("⚠️ No equity data yet — the bot may not have traded.")

    # =======================
    # Trade Log
    # =======================
    st.subheader("📄 Trade History")

    if trades.empty:
        st.info("📭 No trades yet — bot is waiting for valid entry signals.")
    else:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])

        # Filters
        colA, colB = st.columns(2)

        symbol_filter = colA.selectbox("Filter by Symbol", ["All"] + sorted(trades["symbol"].unique()))
        status_filter = colB.selectbox("Filter by Status", ["All", "open", "closed"])

        filtered = trades.copy()
        if symbol_filter != "All":
            filtered = filtered[filtered["symbol"] == symbol_filter]
        if status_filter != "All":
            filtered = filtered[filtered["status"] == status_filter]

        st.dataframe(filtered.reset_index(drop=True))


# ===============================
# 🏁 Run
# ===============================

if __name__ == "__main__":
    main()
