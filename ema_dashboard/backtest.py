from __future__ import annotations

"""Backtesting utilities for EMA crossover strategy variants."""

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from .patterns import prepare_day_df


def _ema(values, period: int):
    """EMA adapter for `backtesting.py` indicator API."""
    return pd.Series(values).ewm(span=period, adjust=False).mean().values


def _safe_number(stats: pd.Series, key: str, default: float = 0.0) -> float:
    """Safely read numeric values from backtest stats."""
    val = stats.get(key, default)
    try:
        if pd.isna(val):
            return default
    except TypeError:
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _variation_score(stats: pd.Series) -> float:
    """Composite ranking score for EMA variation scan."""
    ret = _safe_number(stats, "Return [%]")
    sharpe = _safe_number(stats, "Sharpe Ratio")
    win_rate = _safe_number(stats, "Win Rate [%]")
    max_dd = abs(_safe_number(stats, "Max. Drawdown [%]"))
    trades = max(0.0, _safe_number(stats, "# Trades"))
    trade_confidence = min(1.0, trades / 20.0)
    raw_score = ret + (20.0 * sharpe) + (0.5 * win_rate) - (0.7 * max_dd)
    return raw_score * trade_confidence


class EmaCrossWithEma20Exit(Strategy):
    """Entry on EMA cross; exit when price crosses EMA20 opposite to position."""
    fast = 10
    slow = 20

    def init(self):
        self.ema_fast = self.I(_ema, self.data.Close, self.fast)
        self.ema_slow = self.I(_ema, self.data.Close, self.slow)

    def next(self):
        close = self.data.Close[-1]
        ema20 = self.ema_slow[-1]

        # Exit on opposite side of EMA20.
        if self.position.is_long and close < ema20:
            self.position.close()
        if self.position.is_short and close > ema20:
            self.position.close()

        # Entry on EMA10/EMA20 crossover.
        if crossover(self.ema_fast, self.ema_slow):
            if self.position.is_short:
                self.position.close()
            if not self.position:
                self.buy()
        elif crossover(self.ema_slow, self.ema_fast):
            if self.position.is_long:
                self.position.close()
            if not self.position:
                self.sell()


def run_ema20_exit_backtest(
    df: pd.DataFrame,
    cash: float = 100000,
    commission: float = 0.0005,
    exclusive_orders: bool = True,
    fast: int = 10,
    slow: int = 20,
):
    """Run a single backtest for a selected fast/slow EMA pair."""
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    bt = Backtest(
        df.copy(),
        EmaCrossWithEma20Exit,
        cash=cash,
        commission=commission,
        exclusive_orders=exclusive_orders,
    )
    stats = bt.run(fast=int(fast), slow=int(slow))
    return bt, stats


def run_ema_variation_backtests(
    df: pd.DataFrame,
    fast_periods: list[int],
    slow_periods: list[int],
    cash: float = 100000,
    commission: float = 0.0005,
    exclusive_orders: bool = True,
) -> pd.DataFrame:
    """Grid-scan EMA combinations and return ranked summary table."""
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    fast_unique = sorted({int(x) for x in fast_periods if int(x) > 0})
    slow_unique = sorted({int(x) for x in slow_periods if int(x) > 0})
    if not fast_unique or not slow_unique:
        raise ValueError("fast_periods and slow_periods must have at least one positive value.")

    rows: list[dict[str, float | int]] = []
    for fast in fast_unique:
        for slow in slow_unique:
            if fast >= slow:
                continue

            # Create a temporary strategy class with per-run EMA params.
            class _ParameterizedEmaCross(EmaCrossWithEma20Exit):
                pass

            _ParameterizedEmaCross.fast = fast
            _ParameterizedEmaCross.slow = slow
            bt = Backtest(
                df.copy(),
                _ParameterizedEmaCross,
                cash=cash,
                commission=commission,
                exclusive_orders=exclusive_orders,
            )
            stats = bt.run()
            rows.append(
                {
                    "Fast EMA": fast,
                    "Slow EMA": slow,
                    "Score": _variation_score(stats),
                    "Return [%]": _safe_number(stats, "Return [%]"),
                    "Sharpe Ratio": _safe_number(stats, "Sharpe Ratio"),
                    "Win Rate [%]": _safe_number(stats, "Win Rate [%]"),
                    "Max. Drawdown [%]": _safe_number(stats, "Max. Drawdown [%]"),
                    "# Trades": _safe_number(stats, "# Trades"),
                    "Equity Final [$]": _safe_number(stats, "Equity Final [$]"),
                }
            )

    if not rows:
        raise ValueError("No valid EMA pair generated. Keep fast EMA smaller than slow EMA.")

    result = pd.DataFrame(rows)
    result = result.sort_values(by=["Score", "Return [%]", "Sharpe Ratio"], ascending=[False, False, False]).reset_index(drop=True)
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result


def _max_drawdown_pct(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max.replace(0, np.nan)
    return float(dd.min() * 100.0) if not dd.empty else 0.0


def _sharpe_from_equity(equity: pd.Series) -> float:
    if equity is None or len(equity) < 3:
        return 0.0
    rets = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty:
        return 0.0
    std = float(rets.std())
    if std <= 1e-12:
        return 0.0
    return float((rets.mean() / std) * np.sqrt(len(rets)))


def _prepare_str1_df(
    df: pd.DataFrame,
    min_body_size: float,
    wick_ratio: float,
    prev_body_min_pts: float,
    strategy_settings: dict,
) -> pd.DataFrame:
    day_results = []
    for _, day_df in df.groupby(df.index.date):
        if day_df.empty:
            continue
        day_results.append(
            prepare_day_df(
                day_df,
                min_body_size=min_body_size,
                wick_ratio=wick_ratio,
                prev_body_min_pts=prev_body_min_pts,
                strategy_settings=strategy_settings,
            )
        )
    if not day_results:
        return pd.DataFrame()
    return pd.concat(day_results).sort_index()


def run_str1_signal_backtest(
    df: pd.DataFrame,
    cash: float = 100000,
    commission: float = 0.0005,
) -> tuple[None, pd.Series]:
    """Run backtest from precomputed STR1 Signal/ExitSignal columns."""
    required = {"Open", "High", "Low", "Close", "Signal", "ExitSignal"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required STR1 columns: {sorted(missing)}")

    data = df.copy().sort_index()
    entry_time = None
    entry_price = np.nan
    side = 0
    trade_id = 0
    realized_pnl = 0.0
    trades: list[dict] = []
    equity_rows: list[dict] = []

    for ts, row in data.iterrows():
        close = float(row["Close"])
        signal = str(row.get("Signal", "") or "")
        exit_signal = str(row.get("ExitSignal", "") or "")
        exit_reason = str(row.get("ExitReason", "") or "")

        if side == 0:
            if signal == "B":
                side = 1
                entry_time = ts
                entry_price = close
                realized_pnl -= abs(entry_price) * commission
            elif signal == "S":
                side = -1
                entry_time = ts
                entry_price = close
                realized_pnl -= abs(entry_price) * commission
        else:
            should_exit = (side == 1 and exit_signal == "LX") or (side == -1 and exit_signal == "SX")
            if should_exit:
                trade_id += 1
                pnl = (close - entry_price) * side
                pnl -= abs(close) * commission
                realized_pnl += pnl
                trades.append(
                    {
                        "TradeID": trade_id,
                        "EntryTime": entry_time,
                        "ExitTime": ts,
                        "EntryPrice": float(entry_price),
                        "ExitPrice": close,
                        "Size": float(side),
                        "PnL [$]": float(round(pnl, 4)),
                        "Return [%]": float(round(((close / entry_price) - 1.0) * 100.0 * side, 4)),
                        "Tag": exit_reason if exit_reason else ("LX" if side == 1 else "SX"),
                    }
                )
                side = 0
                entry_time = None
                entry_price = np.nan

        unrealized = 0.0 if side == 0 else (close - float(entry_price)) * float(side)
        equity_rows.append({"Time": ts, "Equity": float(cash + realized_pnl + unrealized)})

    if side != 0 and entry_time is not None:
        ts = data.index[-1]
        close = float(data["Close"].iloc[-1])
        trade_id += 1
        pnl = (close - entry_price) * side
        pnl -= abs(close) * commission
        realized_pnl += pnl
        trades.append(
            {
                "TradeID": trade_id,
                "EntryTime": entry_time,
                "ExitTime": ts,
                "EntryPrice": float(entry_price),
                "ExitPrice": close,
                "Size": float(side),
                "PnL [$]": float(round(pnl, 4)),
                "Return [%]": float(round(((close / entry_price) - 1.0) * 100.0 * side, 4)),
                "Tag": "EOD",
            }
        )
        equity_rows[-1]["Equity"] = float(cash + realized_pnl)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        trades_df = pd.DataFrame(columns=["TradeID", "EntryTime", "ExitTime", "EntryPrice", "ExitPrice", "Size", "PnL [$]", "Return [%]", "Tag"])
    eq_curve = pd.DataFrame(equity_rows).set_index("Time") if equity_rows else pd.DataFrame(columns=["Equity"])
    equity_final = float(eq_curve["Equity"].iloc[-1]) if not eq_curve.empty else float(cash)
    total_return = ((equity_final / float(cash)) - 1.0) * 100.0 if cash else 0.0
    wins = int((trades_df["PnL [$]"] > 0).sum()) if "PnL [$]" in trades_df.columns else 0
    num_trades = int(len(trades_df))
    win_rate = (wins / num_trades * 100.0) if num_trades else 0.0
    max_dd = _max_drawdown_pct(eq_curve["Equity"]) if "Equity" in eq_curve.columns else 0.0
    sharpe = _sharpe_from_equity(eq_curve["Equity"]) if "Equity" in eq_curve.columns else 0.0

    stats = pd.Series(
        {
            "Equity Final [$]": equity_final,
            "Return [%]": total_return,
            "# Trades": float(num_trades),
            "Win Rate [%]": win_rate,
            "Max. Drawdown [%]": max_dd,
            "Sharpe Ratio": sharpe,
            "_strategy": "STR1_Continuation",
            "_equity_curve": eq_curve,
            "_trades": trades_df,
        }
    )
    return None, stats


def run_str1_variation_backtests(
    df: pd.DataFrame,
    min_body_size: float,
    wick_ratio: float,
    prev_body_min_pts: float,
    base_settings: dict,
    fib_options: list[bool],
    zone_options: list[bool],
    atr_options: list[bool],
    atr_len_values: list[int],
    atr_mult_values: list[float],
    cash: float = 100000,
    commission: float = 0.0005,
) -> pd.DataFrame:
    """Grid-scan STR1 exit-criteria combinations."""
    required = {"Open", "High", "Low", "Close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing OHLC columns: {sorted(missing)}")

    rows: list[dict] = []
    fib_unique = [bool(x) for x in fib_options]
    zone_unique = [bool(x) for x in zone_options]
    atr_unique = [bool(x) for x in atr_options]
    atr_len_unique = sorted({int(x) for x in atr_len_values if int(x) > 0}) or [14]
    atr_mult_unique = sorted({float(x) for x in atr_mult_values if float(x) > 0}) or [2.0]

    for fib_on in fib_unique:
        for zone_on in zone_unique:
            for atr_on in atr_unique:
                len_grid = atr_len_unique if atr_on else [int(base_settings.get("atr_len", 14))]
                mult_grid = atr_mult_unique if atr_on else [float(base_settings.get("atr_mult", 2.0))]
                for atr_len in len_grid:
                    for atr_mult in mult_grid:
                        settings = dict(base_settings)
                        settings["exit_fib_enabled"] = fib_on
                        settings["exit_zone_enabled"] = zone_on
                        settings["exit_atr_enabled"] = atr_on
                        settings["atr_len"] = int(atr_len)
                        settings["atr_mult"] = float(atr_mult)
                        prepared = _prepare_str1_df(df, min_body_size, wick_ratio, prev_body_min_pts, settings)
                        if prepared.empty:
                            continue
                        _, stats = run_str1_signal_backtest(prepared, cash=cash, commission=commission)
                        rows.append(
                            {
                                "Fib Exit": int(fib_on),
                                "Zone Exit": int(zone_on),
                                "AtrExit": int(atr_on),
                                "ATR Len": int(atr_len),
                                "ATR Mult": float(atr_mult),
                                "Score": _variation_score(stats),
                                "Return [%]": _safe_number(stats, "Return [%]"),
                                "Sharpe Ratio": _safe_number(stats, "Sharpe Ratio"),
                                "Win Rate [%]": _safe_number(stats, "Win Rate [%]"),
                                "Max. Drawdown [%]": _safe_number(stats, "Max. Drawdown [%]"),
                                "# Trades": _safe_number(stats, "# Trades"),
                                "Equity Final [$]": _safe_number(stats, "Equity Final [$]"),
                            }
                        )

    if not rows:
        raise ValueError("No STR1 variation combination generated.")

    result = pd.DataFrame(rows)
    result = result.sort_values(by=["Score", "Return [%]", "Sharpe Ratio"], ascending=[False, False, False]).reset_index(drop=True)
    result.insert(0, "Rank", range(1, len(result) + 1))
    return result
