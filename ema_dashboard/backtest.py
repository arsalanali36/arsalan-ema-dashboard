from __future__ import annotations

"""Backtesting utilities for EMA crossover strategy variants."""

import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


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
