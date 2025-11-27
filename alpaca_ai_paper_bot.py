import os
import time
import smtplib
import requests
from email.mime.text import MIMEText
from datetime import datetime, date

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from alpaca_trade_api import REST, TimeFrame
from dotenv import load_dotenv


# ==============================
# CONFIG
# ==============================

# Load .env for keys and settings
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# ---- Trading universe (US stocks supported by Alpaca) ----
SYMBOLS = ["AAPL", "MSFT", "GOOGL"]  # you can change/add

DATA_LOOKBACK_DAYS = 365 * 3        # 3 years of history for model
TRAIN_RATIO = 0.7
PROB_THRESHOLD = 0.60               # min prob to go long

STOP_LOSS_PCT = 0.02                # 2% SL
TAKE_PROFIT_PCT = 0.04              # 4% TP
RUN_INTERVAL_SECONDS = 60 * 5       # run every 5 minutes

TRADES_CSV = "trades_log.csv"
EQUITY_CSV = "equity_log.csv"

INITIAL_EQUITY_ASSUMED = 10000.0    # used only if equity log is empty
RISK_PER_TRADE_PCT = 0.02           # risk 2% of equity per trade

# ---- Telegram (from .env) ----
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "0") == "1"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ---- Email (optional) ----
EMAIL_ENABLED = False
EMAIL_FROM = "you@example.com"
EMAIL_TO = "you@example.com"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "you@example.com"
SMTP_PASS = "YOUR_APP_PASSWORD"

# ---- Risk / Alerts ----
RISK_ALERT_DRAWDOWN_PCT = 0.05          # 5% from peak equity
HEARTBEAT_INTERVAL_SECONDS = 60 * 60    # 1 hour
DAILY_SUMMARY_UTC_HOUR = 21             # 21:00 UTC ~ 2:30am IST

# Globals for alert state
LAST_HEARTBEAT_TS = 0.0
LAST_SUMMARY_DATE = None
RISK_ALERT_TRIGGERED = False

print(
    "DEBUG Alpaca config:",
    "base_url =", ALPACA_BASE_URL,
    "| key set =", bool(ALPACA_API_KEY),
    "| secret set =", bool(ALPACA_SECRET_KEY),
    "| telegram =", TELEGRAM_ENABLED,
)


# ==============================
# NOTIFICATIONS
# ==============================

def send_telegram_message(text: str):
    if not TELEGRAM_ENABLED or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown"  # simple formatting
        }
        requests.post(url, data=payload, timeout=5)
    except Exception as e:
        print("Telegram error:", e)


def send_email(subject: str, body: str):
    if not EMAIL_ENABLED:
        return
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM
        msg["To"] = EMAIL_TO

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
    except Exception as e:
        print("Email error:", e)


# ==============================
# INDICATORS & FEATURES
# ==============================

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["Return"] = df["Close"].pct_change()

    df["SMA10"] = df["Close"].rolling(window=10).mean()
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    df["RSI14"] = compute_rsi(df["Close"], period=14)
    df["Volatility10"] = df["Return"].rolling(window=10).std()

    # Future 3-day return (to reduce noise)
    df["FutureReturn3"] = df["Close"].shift(-3) / df["Close"] - 1
    df["Target"] = (df["FutureReturn3"] > 0.003).astype(int)  # >0.3% in next 3 days

    df = df.dropna()
    return df


FEATURE_COLS = ["Return", "SMA10", "SMA20", "SMA50", "RSI14", "Volatility10"]


# ==============================
# MODEL / SIGNAL
# ==============================

def train_model_and_signal(df: pd.DataFrame, prob_threshold=0.6):
    X = df[FEATURE_COLS].values
    y = df["Target"].values

    split_idx = int(len(df) * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Test accuracy for info
    prob_up_test = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_test = (prob_up_test > prob_threshold).astype(int)
    acc = accuracy_score(y_test, y_pred_test)

    # Latest bar
    X_latest_scaled = scaler.transform(X[-1:].copy())
    prob_up_latest = model.predict_proba(X_latest_scaled)[:, 1][0]
    latest_signal = int(prob_up_latest > prob_threshold)

    latest_row = df.iloc[-1].copy()

    return prob_up_latest, latest_signal, latest_row, acc


# ==============================
# ALPACA + TRADE LOGGING
# ==============================

class AlpacaPaperBroker:
    def __init__(self,
                 api_key=ALPACA_API_KEY,
                 secret_key=ALPACA_SECRET_KEY,
                 base_url=ALPACA_BASE_URL,
                 trades_csv=TRADES_CSV,
                 equity_csv=EQUITY_CSV):

        self.api = REST(api_key, secret_key, base_url=base_url)
        self.trades_csv = trades_csv
        self.equity_csv = equity_csv

        # ensure CSVs exist
        if not os.path.exists(self.trades_csv):
            pd.DataFrame(columns=[
                "timestamp", "symbol", "side", "qty", "entry_price",
                "stop_loss", "take_profit", "status", "exit_price",
                "pnl", "reason"
            ]).to_csv(self.trades_csv, index=False)

        if not os.path.exists(self.equity_csv):
            pd.DataFrame(columns=["timestamp", "equity"]).to_csv(self.equity_csv, index=False)

    # ----- CSV helpers -----
    def load_trades(self):
        return pd.read_csv(self.trades_csv)

    def save_trades(self, df):
        df.to_csv(self.trades_csv, index=False)

    def get_open_positions_log(self):
        df = self.load_trades()
        if df.empty:
            return pd.DataFrame(columns=df.columns)
        return df[df["status"] == "open"].copy()

    def current_open_trade_for_symbol(self, symbol):
        opens = self.get_open_positions_log()
        sym = opens[opens["symbol"] == symbol]
        if sym.empty:
            return None
        # latest open trade
        return sym.iloc[-1]

    # ----- Market data via Alpaca -----
    def get_last_price(self, symbol):
        bars = self.api.get_bars(symbol, TimeFrame.Minute, limit=1)
        if len(bars) == 0:
            return None
        return float(bars[0].c)

    # ----- Positions via Alpaca -----
    def alpaca_position_qty(self, symbol):
        try:
            pos = self.api.get_position(symbol)
            return float(pos.qty)
        except Exception:
            return 0.0

    # ----- Orders + logging -----
    def log_new_trade(self, symbol, side, qty, entry_price, stop_loss, take_profit, reason="entry"):
        qty = int(qty)
        ts = datetime.utcnow().isoformat(timespec="seconds")

        # Place Alpaca market order
        if side == "long":
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="gtc"
                )
                print(f"Alpaca BUY sent: id={order.id}, status={order.status}")
            except Exception as e:
                print("❌ Alpaca submit_order BUY FAILED:", e)
                send_telegram_message(f"❌ *BUY FAILED* for `{symbol}`\nReason: `{e}`")
                send_email("Bot Alpaca BUY FAILED", f"{symbol}: {e}")
                return  # don't log trade in CSV if order failed
        else:
            # short selling not implemented here
            return

        # Log to CSV ONLY if order was sent successfully
        df = self.load_trades()
        new_row = {
            "timestamp": ts,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "open",
            "exit_price": np.nan,
            "pnl": np.nan,
            "reason": reason,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.save_trades(df)

        msg = (
            f"🟢 *NEW TRADE OPENED*\n"
            f"• Symbol: `{symbol}`\n"
            f"• Side: *LONG*\n"
            f"• Qty: `{qty}`\n"
            f"• Entry: `${entry_price:.2f}`\n"
            f"• SL: `${stop_loss:.2f}`\n"
            f"• TP: `${take_profit:.2f}`\n"
            f"• Time (UTC): `{ts}`"
        )
        print(msg)
        send_telegram_message(msg)
        send_email("Bot new Alpaca trade", msg)

    def close_trade(self, trade_index, exit_price, reason="exit"):
        df = self.load_trades()
        row = df.loc[trade_index]
        if row["status"] != "open":
            return

        symbol = row["symbol"]
        qty = int(row["qty"])
        side = row["side"]

        # Close via Alpaca
        if side == "long":
            try:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="gtc"
                )
                print(f"Alpaca SELL sent: id={order.id}, status={order.status}")
            except Exception as e:
                print("❌ Alpaca submit_order SELL FAILED:", e)
                send_telegram_message(f"❌ *SELL FAILED* for `{symbol}`\nReason: `{e}`")
                send_email("Bot Alpaca SELL FAILED", f"{symbol}: {e}")
                return

        # pnl calc (approx using our tracked prices)
        if side == "long":
            pnl = (exit_price - row["entry_price"]) * row["qty"]
        else:
            pnl = (row["entry_price"] - exit_price) * row["qty"]

        df.loc[trade_index, "status"] = "closed"
        df.loc[trade_index, "exit_price"] = exit_price
        df.loc[trade_index, "pnl"] = pnl
        df.loc[trade_index, "reason"] = reason
        self.save_trades(df)

        sign = "✅" if pnl >= 0 else "⚠️"
        msg = (
            f"{sign} *TRADE CLOSED* ({reason})\n"
            f"• Symbol: `{symbol}`\n"
            f"• Side: *{side.upper()}*\n"
            f"• Qty: `{qty}`\n"
            f"• Exit: `${exit_price:.2f}`\n"
            f"• PnL: `${pnl:.2f}`"
        )
        print(msg)
        send_telegram_message(msg)
        send_email("Bot closed Alpaca trade", msg)

    def get_equity(self):
        try:
            account = self.api.get_account()
            return float(account.equity)
        except Exception as e:
            print("Error getting equity from Alpaca, using fallback:", e)
            return INITIAL_EQUITY_ASSUMED

    def log_equity(self):
        equity = self.get_equity()
        eq_df = pd.read_csv(self.equity_csv)
        ts = datetime.utcnow().isoformat(timespec="seconds")
        new_row = {"timestamp": ts, "equity": equity}
        eq_df = pd.concat([eq_df, pd.DataFrame([new_row])], ignore_index=True)
        eq_df.to_csv(self.equity_csv, index=False)
        print(f"[EQUITY] {ts} => {equity:.2f}")
        return equity


# ==============================
# RISK / ALERT HELPERS
# ==============================

def check_risk_and_alert():
    """Send a risk alert if equity has fallen more than RISK_ALERT_DRAWDOWN_PCT from peak."""
    global RISK_ALERT_TRIGGERED

    try:
        df = pd.read_csv(EQUITY_CSV)
    except FileNotFoundError:
        return

    if df.empty or "equity" not in df.columns:
        return

    eq = df["equity"].astype(float)
    current_eq = eq.iloc[-1]
    peak_eq = eq.max()
    if peak_eq <= 0:
        return

    drawdown = current_eq / peak_eq - 1.0  # negative number means down from peak

    print(f"DEBUG drawdown: {drawdown * 100:.2f}% from peak")

    if drawdown <= -RISK_ALERT_DRAWDOWN_PCT and not RISK_ALERT_TRIGGERED:
        msg = (
            f"🚨 *RISK ALERT*\n"
            f"Equity dropped `{abs(drawdown) * 100:.2f}%` from peak.\n"
            f"• Peak: `${peak_eq:.2f}`\n"
            f"• Current: `${current_eq:.2f}`"
        )
        send_telegram_message(msg)
        send_email("Bot Risk Alert", msg)
        RISK_ALERT_TRIGGERED = True

    # Optional: reset alert flag if equity recovers near peak
    if drawdown > -RISK_ALERT_DRAWDOWN_PCT / 2:
        RISK_ALERT_TRIGGERED = False


def maybe_send_daily_summary():
    """Send a daily summary once per UTC day after DAILY_SUMMARY_UTC_HOUR."""
    global LAST_SUMMARY_DATE

    if not TELEGRAM_ENABLED:
        return

    now_utc = datetime.utcnow()
    today = now_utc.date()

    # Only after certain hour & only once per day
    if now_utc.hour < DAILY_SUMMARY_UTC_HOUR:
        return
    if LAST_SUMMARY_DATE == today:
        return

    # Load trades & equity
    try:
        trades = pd.read_csv(TRADES_CSV)
    except FileNotFoundError:
        trades = pd.DataFrame()

    try:
        equity = pd.read_csv(EQUITY_CSV)
    except FileNotFoundError:
        equity = pd.DataFrame()

    # Process today's closed trades
    daily_pnl = 0.0
    win_count = 0
    loss_count = 0
    total_closed = 0
    best_trade = 0.0
    worst_trade = 0.0

    if not trades.empty:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
        trades["date"] = trades["timestamp"].dt.date
        closed_today = trades[(trades["status"] == "closed") & (trades["date"] == today)].copy()
        total_closed = len(closed_today)

        if total_closed > 0:
            daily_pnl = float(closed_today["pnl"].sum())
            wins = closed_today[closed_today["pnl"] > 0]
            losses = closed_today[closed_today["pnl"] <= 0]
            win_count = len(wins)
            loss_count = len(losses)
            best_trade = float(closed_today["pnl"].max())
            worst_trade = float(closed_today["pnl"].min())

    end_equity = None
    if not equity.empty:
        end_equity = float(equity["equity"].iloc[-1])

    msg_lines = [
        f"📅 *DAILY SUMMARY* (UTC {today.isoformat()})",
        f"• Closed trades: `{total_closed}`",
        f"• Wins: `{win_count}` | Losses: `{loss_count}`",
        f"• Net PnL: `${daily_pnl:.2f}`",
    ]
    if total_closed > 0:
        msg_lines.append(f"• Best trade: `${best_trade:.2f}`")
        msg_lines.append(f"• Worst trade: `${worst_trade:.2f}`")
    if end_equity is not None:
        msg_lines.append(f"• End Equity: `${end_equity:.2f}`")

    send_telegram_message("\n".join(msg_lines))
    LAST_SUMMARY_DATE = today


def maybe_send_heartbeat():
    """Send periodic 'bot alive' message with latest equity."""
    global LAST_HEARTBEAT_TS

    if not TELEGRAM_ENABLED:
        return

    now_ts = time.time()
    if now_ts - LAST_HEARTBEAT_TS < HEARTBEAT_INTERVAL_SECONDS:
        return

    latest_equity = None
    try:
        eq_df = pd.read_csv(EQUITY_CSV)
        if not eq_df.empty:
            latest_equity = float(eq_df["equity"].iloc[-1])
    except FileNotFoundError:
        pass

    eq_line = f"• Latest equity: `${latest_equity:.2f}`" if latest_equity is not None else "• Equity: N/A"

    msg = (
        f"💓 *BOT HEARTBEAT*\n"
        f"• Time (UTC): `{datetime.utcnow().isoformat(timespec='seconds')}`\n"
        f"{eq_line}"
    )
    send_telegram_message(msg)
    LAST_HEARTBEAT_TS = now_ts


# ==============================
# CORE RUN CYCLE
# ==============================

def run_cycle(symbols):
    print("\n=== RUN CYCLE ===", datetime.utcnow().isoformat(timespec="seconds"))
    broker = AlpacaPaperBroker()

    for symbol in symbols:
        print(f"\n[SYMBOL] {symbol}")

        # 1. Get history via yfinance (for ML features)
        data = yf.download(symbol, period=f"{DATA_LOOKBACK_DAYS}d", interval="1d", progress=False)
        if data.empty:
            print("No data for", symbol)
            continue

        df_feat = add_features(data)
        if len(df_feat) < 100:
            print("Not enough data after feature engineering.")
            continue

        # 2. Train model & get signal
        prob_up, signal, latest_row, acc = train_model_and_signal(df_feat, prob_threshold=PROB_THRESHOLD)

        # For trading decisions, we prefer latest live price from Alpaca
        last_price = broker.get_last_price(symbol)
        if last_price is None:
            last_price = float(latest_row["Close"])

        print(f"Model acc={acc:.2%}, ProbUp={prob_up:.2%}, Signal={signal}, LastPrice={last_price:.2f}")

        # 3. Check if we have an open trade in our log
        open_trade = broker.current_open_trade_for_symbol(symbol)
        alpaca_qty = broker.alpaca_position_qty(symbol)

        # If our log says open but Alpaca has 0 qty -> treat as closed mismatch
        if open_trade is not None and alpaca_qty <= 0:
            broker.close_trade(open_trade.name, exit_price=last_price, reason="sync_mismatch")
            open_trade = None

        # 4. SL/TP handling for open trade
        if open_trade is not None:
            sl = open_trade["stop_loss"]
            tp = open_trade["take_profit"]
            side = open_trade["side"]

            if side == "long":
                if last_price <= sl:
                    broker.close_trade(open_trade.name, exit_price=last_price, reason="stop_loss")
                    open_trade = None
                elif last_price >= tp:
                    broker.close_trade(open_trade.name, exit_price=last_price, reason="take_profit")
                    open_trade = None

        # 5. Entry / exit by signal
        if signal == 1 and open_trade is None:
            # New long
            equity = broker.get_equity()
            risk_amount = equity * RISK_PER_TRADE_PCT
            sl_price = last_price * (1 - STOP_LOSS_PCT)
            tp_price = last_price * (1 + TAKE_PROFIT_PCT)
            per_share_risk = last_price - sl_price
            if per_share_risk <= 0:
                print("Bad SL config; skipping entry.")
                continue

            qty = max(int(risk_amount / per_share_risk), 1)

            broker.log_new_trade(
                symbol=symbol,
                side="long",
                qty=qty,
                entry_price=last_price,
                stop_loss=sl_price,
                take_profit=tp_price,
                reason="signal_long"
            )

        elif signal == 0 and open_trade is not None:
            # Flat signal -> close position
            broker.close_trade(open_trade.name, exit_price=last_price, reason="signal_exit")

    # 6. Log equity after processing all symbols
    broker.log_equity()

    # 7. Risk alert, daily summary
    check_risk_and_alert()
    maybe_send_daily_summary()


# ==============================
# SCHEDULER LOOP
# ==============================

def run_scheduler():
    print("Starting Alpaca AI paper bot scheduler...")
    while True:
        try:
            run_cycle(SYMBOLS)
        except Exception as e:
            print("Error in run_cycle:", e)
            send_email("Bot error", str(e))
            send_telegram_message(f"❌ *BOT ERROR*:\n`{e}`")
        maybe_send_heartbeat()
        print(f"Sleeping for {RUN_INTERVAL_SECONDS} seconds...\n")
        time.sleep(RUN_INTERVAL_SECONDS)


if __name__ == "__main__":
    # For testing one cycle:
    # run_cycle(SYMBOLS)

    # Continuous paper trading:
    run_scheduler()
