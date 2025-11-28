import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from sqlalchemy import create_engine
import os

# ========== DATABASE CONNECTION ==========
DATABASE_URL = os.getenv("DATABASE_URL")  # make sure this is set in Render environment
engine = create_engine(DATABASE_URL)


# ========== DB READ FUNCTIONS ==========

def load_trades():
    try:
        return pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC", engine)
    except Exception as e:
        st.error(f"❌ Error loading trades from DB: {e}")
        return pd.DataFrame(columns=[
            "timestamp","symbol","side","qty","entry_price",
            "stop_loss","take_profit","status","exit_price",
            "pnl","reason"
        ])


def load_equity():
    try:
        return pd.read_sql("SELECT * FROM equity ORDER BY timestamp ASC", engine)
    except Exception as e:
        st.error(f"❌ Error loading equity from DB: {e}")
        return pd.DataFrame(columns=["timestamp", "equity"])



# ========== DASHBOARD UI ==========

def main():
    st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

    st.title("📈 Live Trading Dashboard (Alpaca + AI Bot)")

    st.write("""
    This dashboard shows live updates from the **AI trading bot** running on Railway.

    - ✔ Auto-trade execution
    - ✔ Stop-loss / Take-profit logic
    - ✔ Telegram notifications
    - ✔ Portfolio tracking  
    """)

    trades = load_trades()
    equity = load_equity()

    # ===== Summary =====
    st.subheader("📊 Performance Summary")

    if not equity.empty:
        equity["timestamp"] = pd.to_datetime(equity["timestamp"])
        latest_equity = equity["equity"].iloc[-1]
        starting_equity = equity["equity"].iloc[0]
        total_return = (latest_equity / starting_equity - 1) * 100
    else:
        latest_equity = 0
        total_return = 0

    closed_trades = trades[trades["status"] == "closed"] if not trades.empty else pd.DataFrame()
    wins = len(closed_trades[closed_trades["pnl"] > 0]) if not closed_trades.empty else 0
    total_closed = len(closed_trades)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Equity", f"${latest_equity:,.2f}")
    col2.metric("📈 Total Return", f"{total_return:.2f}%")
    col3.metric("📉 Total Trades", len(trades))
    col4.metric("🏆 Win Rate", f"{(wins / total_closed * 100):.2f}%" if total_closed > 0 else "N/A")


    # ===== Equity Chart =====
    st.subheader("📈 Equity Curve")

    if not equity.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(equity["timestamp"], equity["equity"], linewidth=2)

        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No equity data available yet. Bot may not have taken trades.")


    # ===== Trade Log Table =====
    st.subheader("📄 Trade Log")

    if trades.empty:
        st.info("No trades yet — bot may be waiting for signals.")
    else:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
        
        # Filters
        colA, colB = st.columns(2)

        symbol_filter = colA.selectbox("Filter by symbol", ["All"] + sorted(trades["symbol"].unique().tolist()))
        status_filter = colB.selectbox("Filter by status", ["All", "open", "closed"])

        filtered = trades.copy()
        if symbol_filter != "All":
            filtered = filtered[filtered["symbol"] == symbol_filter]
        if status_filter != "All":
            filtered = filtered[filtered["status"] == status_filter]

        st.dataframe(filtered.reset_index(drop=True))


if __name__ == "__main__":
    main()
