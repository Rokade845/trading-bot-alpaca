import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
import matplotlib.dates as mdates

TRADES_CSV = "trades_log.csv"
EQUITY_CSV = "equity_log.csv"


def load_trades():
    try:
        return pd.read_csv(TRADES_CSV)
    except FileNotFoundError:
        return pd.DataFrame(columns=[
            "timestamp", "symbol", "side", "qty", "entry_price",
            "stop_loss", "take_profit", "status", "exit_price",
            "pnl", "reason"
        ])


def load_equity():
    try:
        return pd.read_csv(EQUITY_CSV)
    except FileNotFoundError:
        return pd.DataFrame(columns=["timestamp", "equity"])


def compute_max_drawdown(equity_series: pd.Series) -> float:
    """Returns max drawdown as a negative percentage (e.g. -0.15 = -15%)."""
    if equity_series.empty:
        return 0.0
    cum_max = equity_series.cummax()
    dd = equity_series / cum_max - 1.0
    return dd.min()


def compute_sharpe_from_equity(equity_df: pd.DataFrame) -> float:
    """
    Rough Sharpe using equity changes as returns.
    Assumes each step is roughly '1 period' (not annualized properly, but ok as relative metric).
    """
    if equity_df.empty or len(equity_df) < 3:
        return 0.0
    eq = equity_df["equity"].astype(float)
    rets = eq.pct_change().dropna()
    if rets.std() == 0:
        return 0.0
    # Simple (not annualized strictly): mean / std * sqrt(N)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(len(rets))
    return sharpe


def main():
    st.set_page_config(page_title="Alpaca AI Paper Trading Dashboard", layout="wide")

    st.title("🤖 Alpaca AI Paper Trading Dashboard")

    col_top1, col_top2 = st.columns(2)
    with col_top1:
        st.markdown(
            """
            This dashboard shows activity from your **Alpaca AI paper bot**:

            - 📈 Equity curve over time  
            - 🧾 Trade log (open & closed)  
            - 🎯 Win-rate, PnL, drawdown  
            """
        )

    with col_top2:
        st.info("Make sure `alpaca_ai_paper_bot.py` is running in another terminal.")

    trades = load_trades()
    equity = load_equity()

    # ===== Pre-process equity =====
    if not equity.empty:
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], errors="coerce")
        equity = equity.dropna(subset=["timestamp", "equity"])
        equity = equity.sort_values("timestamp")
        latest_equity = float(equity["equity"].iloc[-1])
        start_equity = float(equity["equity"].iloc[0])
        total_return = (latest_equity / start_equity - 1) * 100 if start_equity > 0 else 0.0
        max_dd = compute_max_drawdown(equity["equity"])
        sharpe = compute_sharpe_from_equity(equity)
    else:
        latest_equity = 0.0
        total_return = 0.0
        max_dd = 0.0
        sharpe = 0.0

    # ===== Pre-process trades =====
    if not trades.empty:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
        trades = trades.sort_values("timestamp", ascending=False)
        closed = trades[trades["status"] == "closed"].copy()
        open_trades = trades[trades["status"] == "open"].copy()

        total_trades = len(trades)
        closed_trades = len(closed)

        total_pnl = float(closed["pnl"].sum()) if not closed.empty else 0.0

        wins_df = closed[closed["pnl"] > 0] if not closed.empty else pd.DataFrame()
        losses_df = closed[closed["pnl"] <= 0] if not closed.empty else pd.DataFrame()

        wins = len(wins_df)
        losses = len(losses_df)

        avg_win = float(wins_df["pnl"].mean()) if not wins_df.empty else 0.0
        avg_loss = float(losses_df["pnl"].mean()) if not losses_df.empty else 0.0

        win_rate = (wins / closed_trades * 100) if closed_trades > 0 else 0.0
        best_trade = float(closed["pnl"].max()) if not closed.empty else 0.0
        worst_trade = float(closed["pnl"].min()) if not closed.empty else 0.0
    else:
        closed = pd.DataFrame()
        open_trades = pd.DataFrame()
        total_trades = 0
        closed_trades = 0
        total_pnl = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        win_rate = 0.0
        best_trade = 0.0
        worst_trade = 0.0

    # ===== Summary =====
    st.subheader("📊 Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Equity", f"{latest_equity:,.2f}")
    c2.metric("Total Return %", f"{total_return:.2f}%")
    c3.metric("Total PnL (closed)", f"{total_pnl:,.2f}")
    c4.metric("Win Rate", f"{win_rate:.2f}%" if closed_trades > 0 else "N/A")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Max Drawdown", f"{max_dd * 100:.2f}%")
    c6.metric("Sharpe (rough)", f"{sharpe:.2f}")
    c7.metric("Avg Win", f"{avg_win:,.2f}")
    c8.metric("Avg Loss", f"{avg_loss:,.2f}")

    # ===== Tabs =====
    tab1, tab2, tab3 = st.tabs(["📈 Equity Curve", "🧾 Trades Log", "📉 PnL Distribution"])

    # ---------- TAB 1: Equity Curve ----------
    with tab1:
        st.subheader("Equity Curve")

        if not equity.empty:
            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(equity["timestamp"], equity["equity"], linewidth=2)

            # Dynamic time axis formatting
            time_span = equity["timestamp"].max() - equity["timestamp"].min()
            if time_span.days >= 2:
                # Show dates if spanning multiple days
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            elif time_span.days >= 1:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())

            plt.xticks(rotation=45)
            plt.tight_layout()

            ax.set_xlabel("Time")
            ax.set_ylabel("Equity ($)")
            ax.set_title("Equity Curve (Paper Trading)")

            st.pyplot(fig)
            plt.close(fig)
        else:
            st.write("No equity data yet. Let the bot run for a while.")

    # ---------- TAB 2: Trades Log ----------
    with tab2:
        st.subheader("Trades Log")

        if trades.empty:
            st.write("No trades logged yet.")
        else:
            # Filters
            symbols = ["All"] + sorted(trades["symbol"].dropna().unique().tolist())
            selected_symbol = st.selectbox("Filter by symbol", symbols)

            statuses = ["All", "open", "closed"]
            selected_status = st.selectbox("Filter by status", statuses)

            reasons = ["All"] + sorted(trades["reason"].dropna().unique().tolist())
            selected_reason = st.selectbox("Filter by reason", reasons)

            filtered = trades.copy()
            if selected_symbol != "All":
                filtered = filtered[filtered["symbol"] == selected_symbol]
            if selected_status != "All":
                filtered = filtered[filtered["status"] == selected_status]
            if selected_reason != "All":
                filtered = filtered[filtered["reason"] == selected_reason]

            st.markdown("### Filtered Trades")
            st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

            col_open, col_closed = st.columns(2)
            with col_open:
                st.markdown("#### Open Positions")
                if open_trades.empty:
                    st.write("No open trades.")
                else:
                    st.dataframe(open_trades.reset_index(drop=True), use_container_width=True)

            with col_closed:
                st.markdown("#### Closed Trades (PnL)")
                if closed.empty:
                    st.write("No closed trades yet.")
                else:
                    st.dataframe(closed.sort_values("timestamp", ascending=False).reset_index(drop=True),
                                 use_container_width=True)

            st.download_button(
                "⬇️ Download all trades as CSV",
                data=trades.to_csv(index=False),
                file_name="trades_log.csv",
                mime="text/csv",
            )

    # ---------- TAB 3: PnL Distribution ----------
    with tab3:
        st.subheader("PnL Distribution (Closed Trades)")

        if closed.empty:
            st.write("No closed trades yet.")
        else:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.hist(closed["pnl"], bins=20)
            ax2.set_xlabel("Trade PnL")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Distribution of Trade PnL (Closed Trades)")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)


if __name__ == "__main__":
    main()
