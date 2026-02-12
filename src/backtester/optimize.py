from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Callable
import itertools
import math
import random

import numpy as np
import pandas as pd

from data import BMCEDataSource, YahooFinanceDataSource, MarketData
from strategy import SignalFrame
from indicators import IndicatorEngine, FeatureSpec, FeaturesData
from engine import EngineSpec, DataConfig, StrategyConfig, estimate_warmup_bars_from_params, BacktestEngine
from portfolio import PortfolioEngine, PortfolioConfig


# ============================================================
# Public dataclasses
# ============================================================

@dataclass(frozen=True)
class OptimizeConfig:
    method: str = "random"        # "random" | "grid"
    seed: int = 42
    n_trials: int = 300           # for random
    top_k: int = 30

    # indicator engine cache options (optimization usually wants memory cache only)
    feature_cache_dir: str = ".cache/features"
    enable_disk_cache: bool = False
    enable_memory_cache: bool = True


@dataclass(frozen=True)
class ParamDef:
    """
    key:
      - "strategy.sma_fast_window", "strategy.sma_slow_window", "strategy.sma_window"
      - "portfolio.cooldown_bars", "portfolio.buy_pct_cash", "portfolio.sell_pct_shares"
      - "data.window" -> tuple(start,end) strings

    kind:
      - "int" | "float" | "choice" | "date_window"

    domain:
      - int:   (lo, hi, step)
      - float: (lo, hi, step)
      - choice: [v1, v2, ...]
      - date_window: [(start, end), ...]
    """
    key: str
    kind: str
    domain: Any
    cast: Callable[[Any], Any] = lambda x: x
    enabled: bool = True


@dataclass(frozen=True)
class TrialResult:
    params: Dict[str, Any]
    pnl: float
    traded_notional: float
    efficiency: float
    n_fills: int
    cagr: float 
    error: Optional[str] = None
    
@dataclass
class BankRequest:
    sma: set[int] = field(default_factory=set)
    rsi: set[int] = field(default_factory=set)
    ema: set[int] = field(default_factory=set)
    macd: set[tuple[int,int,int]] = field(default_factory=set)
    std: set[int] = field(default_factory=set)   # NEW: rolling std windows


    def merge(self, other: "BankRequest") -> "BankRequest":
        self.sma |= other.sma
        self.rsi |= other.rsi
        self.ema |= other.ema
        self.macd |= other.macd
        self.std |= other.std
        return self

def rolling_std_cumsum(close: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(close, np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if w <= 0 or w > n:
        return out

    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))

    sum_x = c1[w:] - c1[:-w]
    sum_x2 = c2[w:] - c2[:-w]

    mean = sum_x / w
    var = (sum_x2 / w) - mean * mean
    var = np.maximum(var, 0.0)  # numerical guard
    out[w-1:] = np.sqrt(var)
    return out


def sma_cumsum(close: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(close, np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if w <= 0 or w > n:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[w-1:] = (c[w:] - c[:-w]) / w
    return out

def ema(x: np.ndarray, span: int) -> np.ndarray:
    a = np.asarray(x, np.float64)
    n = a.size
    out = np.full(n, np.nan, dtype=np.float64)
    if span <= 0 or n == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    # seed at first finite
    i0 = np.argmax(np.isfinite(a))
    if not np.isfinite(a[i0]):
        return out
    out[i0] = a[i0]
    for i in range(i0 + 1, n):
        if np.isfinite(a[i]):
            out[i] = alpha * a[i] + (1 - alpha) * out[i-1]
        else:
            out[i] = out[i-1]
    return out
# ============================================================
# "Today" signal helpers (used for optimizer leaderboard)
# ============================================================

def _signal_to_label(sig: Any) -> str:
    """
    Map numeric signal to label.
    Convention:
      +1 => BUY
      -1 => SELL
       0 => HOLD
      NaN/None => NA
    """
    if sig is None or (isinstance(sig, float) and np.isnan(sig)):
        return "NA"
    try:
        s = int(sig)
    except Exception:
        return "NA"
    if s > 0:
        return "BUY"
    if s < 0:
        return "SELL"
    return "HOLD"


def _extract_params_from_row(row: dict) -> Dict[str, Any]:
    """
    Keep only param-like keys from a leaderboard row.
    (Avoid passing metrics like pnl/cagr/n_fills/error into the adapter.)
    """
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(k, str) and (
            k.startswith("strategy.") or k.startswith("portfolio.") or k.startswith("data.")
        ):
            out[k] = v
    return out


def _latest_signal_for_params(
    *,
    adapter: "StrategyAdapter",
    base_spec: EngineSpec,
    symbols: List[str],
    index: pd.DatetimeIndex,
    bank: Dict[str, Dict[str, np.ndarray]],
    bars_close: Dict[str, np.ndarray],
    params: Dict[str, Any],
) -> Tuple[pd.Timestamp, float, str]:
    """
    Compute last-bar signal using the SAME adapter logic used during optimization.
    Returns: (signal_date, numeric_signal, label)
    """
    signal_date = pd.Timestamp(index[-1])

    sf = adapter.make_signals_from_bank(
        symbols=symbols,
        index=index,
        bank=bank,
        bars_close=bars_close,
        params=params,
        base_spec=base_spec,
    )

    sym0 = symbols[0]
    sig_val = np.nan
    is_valid = True

    if sf.signals is not None and sym0 in sf.signals.columns and len(sf.signals) > 0:
        sig_val = sf.signals[sym0].iloc[-1]

    if sf.validity is not None and sym0 in sf.validity.columns and len(sf.validity) > 0:
        is_valid = bool(sf.validity[sym0].iloc[-1])

    if not is_valid:
        return signal_date, float(sig_val) if sig_val is not None else float("nan"), "NA"

    label = _signal_to_label(sig_val)
    return signal_date, float(sig_val) if sig_val is not None else float("nan"), label

def rsi_wilder(close: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(close, np.float64)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 0 or n < window + 1:
        return out

    delta = np.diff(x)
    up = np.maximum(delta, 0.0)
    dn = np.maximum(-delta, 0.0)

    avg_up = np.full(n-1, np.nan, dtype=np.float64)
    avg_dn = np.full(n-1, np.nan, dtype=np.float64)

    avg_up[window-1] = np.mean(up[:window])
    avg_dn[window-1] = np.mean(dn[:window])

    alpha = 1.0 / window
    for i in range(window, n-1):
        avg_up[i] = (1 - alpha) * avg_up[i-1] + alpha * up[i]
        avg_dn[i] = (1 - alpha) * avg_dn[i-1] + alpha * dn[i]

    rs = avg_up / np.where(avg_dn == 0.0, np.nan, avg_dn)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    out[1:] = rsi
    return out

def macd_pack(close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ef = ema(close, fast)
    es = ema(close, slow)
    macd_line = ef - es
    sig_line = ema(macd_line, signal)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def build_bank(close_by_sym: Dict[str, np.ndarray], req: BankRequest) -> Dict[str, Dict[str, np.ndarray]]:
    bank: Dict[str, Dict[str, np.ndarray]] = {sym: {} for sym in close_by_sym}

    # If MACD requested, we *implicitly* need EMA(close, fast/slow)
    # We can expand req.ema here (optional, but clean).
    if req.macd:
        for (f, s, _g) in req.macd:
            req.ema.add(int(f))
            req.ema.add(int(s))

    for sym, close in close_by_sym.items():
        sym_bank: Dict[str, np.ndarray] = bank[sym]

        # ---- SMA ----
        for w in sorted(req.sma):
            sym_bank[f"sma_{int(w)}"] = sma_cumsum(close, int(w))

        # ---- RSI ----
        for w in sorted(req.rsi):
            sym_bank[f"rsi_{int(w)}"] = rsi_wilder(close, int(w))

        # ---- EMA (explicit requests) ----
        # keep an EMA cache so MACD can reuse it too
        ema_close_cache: Dict[int, np.ndarray] = {}
        for span in sorted(req.ema):
            span = int(span)
            ema_close_cache[span] = ema(close, span)
            sym_bank[f"ema_{span}"] = ema_close_cache[span]  # optional: store

        # ---- MACD (reuses EMA(close, span)) ----
        for (f, s, g) in sorted(req.macd):
            f, s, g = int(f), int(s), int(g)
            m, ms, h = macd_pack_cached(close, f, s, g, ema_close_cache)
            sym_bank[f"macd_{f}_{s}_{g}"] = m
            sym_bank[f"macd_signal_{f}_{s}_{g}"] = ms
            sym_bank[f"macd_hist_{f}_{s}_{g}"] = h
        # ---- Rolling STD ----
        for w in sorted(req.std):
            sym_bank[f"std_{int(w)}"] = rolling_std_cumsum(close, int(w))


    return bank


# ============================================================
# Strategy adapter registry (fast + strategy-agnostic optimization)
# ============================================================

@dataclass(frozen=True)
class StrategyAdapter:
    kind: str

    def required_bank(self, base_spec: EngineSpec, active_params: list[ParamDef]) -> BankRequest:
        return BankRequest()

    def make_signals_from_bank(
        self,
        symbols: List[str],
        index: pd.DatetimeIndex,
        bank: Dict[str, Dict[str, np.ndarray]],
        bars_close: Dict[str, np.ndarray],
        params: Dict[str, Any],
        base_spec: EngineSpec,
    ) -> SignalFrame:
        raise NotImplementedError

    def validate_params(self, params: Dict[str, Any], base_spec: EngineSpec) -> Tuple[bool, Optional[str]]:
        return True, None


class MACrossAdapter(StrategyAdapter):
    def __init__(self):
        super().__init__(kind="ma_cross")

    def validate_params(self, params: Dict[str, Any], base_spec: EngineSpec) -> Tuple[bool, Optional[str]]:
        f = int(params.get("strategy.sma_fast_window", base_spec.strategy.params.get("sma_fast_window", 15)))
        s = int(params.get("strategy.sma_slow_window", base_spec.strategy.params.get("sma_slow_window", 50)))
        if f >= s:
            return False, "fast_window must be < slow_window"
        if f <= 0 or s <= 0:
            return False, "windows must be positive"
        return True, None

    def required_bank(self, base_spec, active_params):
        req = BankRequest()
        f0 = int(base_spec.strategy.params.get("fast_window", 15))
        s0 = int(base_spec.strategy.params.get("slow_window", 50))
        req.sma.add(f0)
        req.sma.add(s0)
        req.sma |= set(_domain_values_int(active_params, "strategy.sma_fast_window"))
        req.sma |= set(_domain_values_int(active_params, "strategy.sma_slow_window"))
        return req


    def make_signals_from_bank(
        self,
        symbols: List[str],
        index: pd.DatetimeIndex,
        bank: Dict[str, Dict[str, np.ndarray]],
        bars_close: Dict[str, np.ndarray],
        params: Dict[str, Any],
        base_spec: EngineSpec,
    ) -> SignalFrame:
        f = int(params.get("strategy.sma_fast_window", base_spec.strategy.params.get("sma_fast_window", 15)))
        s = int(params.get("strategy.sma_slow_window", base_spec.strategy.params.get("sma_slow_window", 50)))

        allow_short = bool(base_spec.strategy.params.get("allow_short", False))
        # allow overriding allow_short if the app exposes it as a choice param
        if "strategy.allow_short" in params:
            allow_short = bool(params["strategy.allow_short"])

        nan_policy = str(base_spec.strategy.params.get("nan_policy", "flat"))
        if "strategy.nan_policy" in params:
            nan_policy = str(params["strategy.nan_policy"])

        col_fast = f"sma_{f}"
        col_slow = f"sma_{s}"

        sig = pd.DataFrame(index=index, columns=symbols, dtype="float64")
        valid = pd.DataFrame(index=index, columns=symbols, dtype="bool")

        for sym in symbols:
            fast = bank[sym][col_fast]
            slow = bank[sym][col_slow]

            v = (~np.isnan(fast)) & (~np.isnan(slow))
            long_mask = fast > slow

            if allow_short:
                short_mask = fast < slow
                out = np.zeros(len(index), dtype=np.float64)
                out[long_mask] = 1.0
                out[short_mask] = -1.0
            else:
                out = long_mask.astype(np.float64)

            if nan_policy == "flat":
                out = np.where(v, out, 0.0)
            else:
                out = np.where(v, out, np.nan)

            sig[sym] = out
            valid[sym] = v

        return SignalFrame(
            signals=sig,
            validity=valid,
            meta={"adapter": "ma_cross", "fast_window": f, "slow_window": s, "allow_short": allow_short, "nan_policy": nan_policy},
        )


class PriceAboveSMAAdapter(StrategyAdapter):
    def __init__(self):
        super().__init__(kind="sma_price")

    def required_bank(self, base_spec: EngineSpec, active_params: List[ParamDef]) -> BankRequest:
        req = BankRequest()
        w0 = int(base_spec.strategy.params.get("window", 50))
        req.sma.add(w0)
        req.sma |= set(_domain_values_int(active_params, "strategy.sma_window"))
        return req


    def validate_params(self, params: Dict[str, Any], base_spec: EngineSpec) -> Tuple[bool, Optional[str]]:
        w = int(params.get("strategy.sma_window", base_spec.strategy.params.get("sma_window", 50)))
        if w <= 0:
            return False, "window must be positive"
        return True, None

    def make_signals_from_bank(
        self,
        symbols: List[str],
        index: pd.DatetimeIndex,
        bank: Dict[str, Dict[str, np.ndarray]],
        bars_close: Dict[str, np.ndarray],
        params: Dict[str, Any],
        base_spec: EngineSpec,
    ) -> SignalFrame:
        w = int(params.get("strategy.sma_window", base_spec.strategy.params.get("sma_window", 50)))

        allow_short = bool(base_spec.strategy.params.get("allow_short", False))
        if "strategy.allow_short" in params:
            allow_short = bool(params["strategy.allow_short"])

        nan_policy = str(base_spec.strategy.params.get("nan_policy", "flat"))
        if "strategy.nan_policy" in params:
            nan_policy = str(params["strategy.nan_policy"])

        col_sma = f"sma_{w}"

        sig = pd.DataFrame(index=index, columns=symbols, dtype="float64")
        valid = pd.DataFrame(index=index, columns=symbols, dtype="bool")

        for sym in symbols:
            close = bars_close[sym]
            sma = bank[sym][col_sma]
            v = (~np.isnan(close)) & (~np.isnan(sma))

            long_mask = close > sma
            if allow_short:
                short_mask = close < sma
                out = np.zeros(len(index), dtype=np.float64)
                out[long_mask] = 1.0
                out[short_mask] = -1.0
            else:
                out = long_mask.astype(np.float64)

            if nan_policy == "flat":
                out = np.where(v, out, 0.0)
            else:
                out = np.where(v, out, np.nan)

            sig[sym] = out
            valid[sym] = v

        return SignalFrame(
            signals=sig,
            validity=valid,
            meta={"adapter": "sma_price", "window": w, "allow_short": allow_short, "nan_policy": nan_policy},
        )

class RSIStrategyAdapter(StrategyAdapter):
    def __init__(self):
        super().__init__(kind="rsi")

    def validate_params(self, params: Dict[str, Any], base_spec: EngineSpec) -> Tuple[bool, Optional[str]]:
        w = int(params.get("strategy.rsi_window", base_spec.strategy.params.get("rsi_window", 14)))
        low = float(params.get("strategy.rsi_oversold", base_spec.strategy.params.get("rsi_oversold", 30.0)))
        high = float(params.get("strategy.rsi_overbought", base_spec.strategy.params.get("rsi_overbought", 70.0)))

        if w <= 0:
            return False, "rsi_window must be positive"
        if not (0.0 <= low <= 100.0 and 0.0 <= high <= 100.0):
            return False, "rsi_oversold/rsi_overbought must be in [0, 100]"
        if low >= high:
            return False, "rsi_oversold must be < rsi_overbought"
        return True, None

    def required_bank(self, base_spec: EngineSpec, active_params: List[ParamDef]) -> BankRequest:
        req = BankRequest()
        w0 = int(base_spec.strategy.params.get("rsi_window", 14))
        req.rsi.add(w0)
        req.rsi |= set(_domain_values_int(active_params, "strategy.rsi_window"))
        return req


    def make_signals_from_bank(
        self,
        symbols: List[str],
        index: pd.DatetimeIndex,
        bank: Dict[str, Dict[str, np.ndarray]],
        bars_close: Dict[str, np.ndarray],
        params: Dict[str, Any],
        base_spec: EngineSpec,
    ) -> SignalFrame:
        w = int(params.get("strategy.rsi_window", base_spec.strategy.params.get("rsi_window", 14)))
        low = float(params.get("strategy.rsi_oversold", base_spec.strategy.params.get("rsi_oversold", 30.0)))
        high = float(params.get("strategy.rsi_overbought", base_spec.strategy.params.get("rsi_overbought", 70.0)))

        allow_short = bool(base_spec.strategy.params.get("allow_short", False))
        if "strategy.allow_short" in params:
            allow_short = bool(params["strategy.allow_short"])

        nan_policy = str(base_spec.strategy.params.get("nan_policy", "flat"))
        if "strategy.nan_policy" in params:
            nan_policy = str(params["strategy.nan_policy"])

        col = f"rsi_{w}"

        sig = pd.DataFrame(index=index, columns=symbols, dtype="float64")
        valid = pd.DataFrame(index=index, columns=symbols, dtype="bool")

        for sym in symbols:
            rsi = bank[sym][col]
            v = ~np.isnan(rsi)

            # Stateless mean-reversion:
            long_mask = rsi < low

            if allow_short:
                short_mask = rsi > high
                out = np.zeros(len(index), dtype=np.float64)
                out[long_mask] = 1.0
                out[short_mask] = -1.0
            else:
                out = long_mask.astype(np.float64)

            if nan_policy == "flat":
                out = np.where(v, out, 0.0)
            else:
                out = np.where(v, out, np.nan)

            sig[sym] = out
            valid[sym] = v

        return SignalFrame(
            signals=sig,
            validity=valid,
            meta={"adapter": "rsi", "rsi_window": w, "rsi_oversold": low, "rsi_overbought": high, "allow_short": allow_short, "nan_policy": nan_policy},
        )

class MACDStrategyAdapter(StrategyAdapter):
    def __init__(self):
        super().__init__(kind="macd")

    def validate_params(self, params: Dict[str, Any], base_spec: EngineSpec) -> Tuple[bool, Optional[str]]:
        f = int(params.get("strategy.macd_fast_window", base_spec.strategy.params.get("macd_fast_window", 12)))
        s = int(params.get("strategy.macd_slow_window", base_spec.strategy.params.get("macd_slow_window", 26)))
        sig = int(params.get("strategy.macd_signal_window", base_spec.strategy.params.get("macd_signal_window", 9)))

        if f <= 0 or s <= 0 or sig <= 0:
            return False, "macd_fast/slow/signal must be positive"
        if f >= s:
            return False, "macd_fast must be < macd_slow"
        return True, None

    def required_bank(self, base_spec: EngineSpec, active_params: List[ParamDef]) -> BankRequest:
        req = BankRequest()

        f0 = int(base_spec.strategy.params.get("macd_fast_window", 12))
        s0 = int(base_spec.strategy.params.get("macd_slow_window", 26))
        g0 = int(base_spec.strategy.params.get("macd_signal_window", 9))

        fs = _domain_values_int(active_params, "strategy.macd_fast_window") or [f0]
        ss = _domain_values_int(active_params, "strategy.macd_slow_window") or [s0]
        gs = _domain_values_int(active_params, "strategy.macd_signal_window") or [g0]

        for f in (fs + [f0]):
            for s in (ss + [s0]):
                for g in (gs + [g0]):
                    f, s, g = int(f), int(s), int(g)
                    if f > 0 and s > 0 and g > 0 and f < s:
                        req.macd.add((f, s, g))

        return req


    def make_signals_from_bank(
        self,
        symbols: List[str],
        index: pd.DatetimeIndex,
        bank: Dict[str, Dict[str, np.ndarray]],
        bars_close: Dict[str, np.ndarray],
        params: Dict[str, Any],
        base_spec: EngineSpec,
    ) -> SignalFrame:
        f = int(params.get("strategy.macd_fast_window", base_spec.strategy.params.get("macd_fast_window", 12)))
        s = int(params.get("strategy.macd_slow_window", base_spec.strategy.params.get("macd_slow_window", 26)))
        g = int(params.get("strategy.macd_signal_window", base_spec.strategy.params.get("macd_signal_window", 9)))

        use_hist = bool(base_spec.strategy.params.get("macd_use_hist", False))
        if "strategy.macd_use_hist" in params:
            use_hist = bool(params["strategy.macd_use_hist"])

        allow_short = bool(base_spec.strategy.params.get("allow_short", False))
        if "strategy.allow_short" in params:
            allow_short = bool(params["strategy.allow_short"])

        nan_policy = str(base_spec.strategy.params.get("nan_policy", "flat"))
        if "strategy.nan_policy" in params:
            nan_policy = str(params["strategy.nan_policy"])

        col_macd = f"macd_{f}_{s}_{g}"          # MACD line
        col_sig  = f"macd_signal_{f}_{s}_{g}"   # signal line
        col_hist = f"macd_hist_{f}_{s}_{g}"     # histogram

        sig_df = pd.DataFrame(index=index, columns=symbols, dtype="float64")
        valid_df = pd.DataFrame(index=index, columns=symbols, dtype="bool")

        for sym in symbols:
            if use_hist:
                hist = bank[sym][col_hist]
                v = ~np.isnan(hist)
                long_mask = hist > 0
                short_mask = hist < 0
            else:
                macd = bank[sym][col_macd]
                msig = bank[sym][col_sig]
                v = (~np.isnan(macd)) & (~np.isnan(msig))
                long_mask = macd > msig
                short_mask = macd < msig

            if allow_short:
                out = np.zeros(len(index), dtype=np.float64)
                out[long_mask] = 1.0
                out[short_mask] = -1.0
            else:
                out = long_mask.astype(np.float64)

            if nan_policy == "flat":
                out = np.where(v, out, 0.0)
            else:
                out = np.where(v, out, np.nan)

            sig_df[sym] = out
            valid_df[sym] = v

        return SignalFrame(
            signals=sig_df,
            validity=valid_df,
            meta={"adapter": "macd", "fast": f, "slow": s, "signal": g, "use_hist": use_hist, "allow_short": allow_short, "nan_policy": nan_policy},
        )

class BollingerAdapter(StrategyAdapter):
    def __init__(self):
        super().__init__(kind="bollinger")

    def validate_params(self, params: Dict[str, Any], base_spec: EngineSpec) -> Tuple[bool, Optional[str]]:
        w = int(params.get("strategy.bb_window", base_spec.strategy.params.get("bb_window", 20)))
        k = float(params.get("strategy.bb_k", base_spec.strategy.params.get("bb_k", 2.0)))
        if w <= 1:
            return False, "bb_window must be > 1"
        if k <= 0:
            return False, "bb_k must be positive"
        return True, None

    def required_bank(self, base_spec: EngineSpec, active_params: List[ParamDef]) -> BankRequest:
        req = BankRequest()
        w0 = int(base_spec.strategy.params.get("bb_window", 20))
        req.sma.add(w0)
        req.std.add(w0)

        ws = _domain_values_int(active_params, "strategy.bb_window")
        for w in ws:
            if w is not None and int(w) > 1:
                req.sma.add(int(w))
                req.std.add(int(w))
        return req

    def make_signals_from_bank(
        self,
        symbols: List[str],
        index: pd.DatetimeIndex,
        bank: Dict[str, Dict[str, np.ndarray]],
        bars_close: Dict[str, np.ndarray],
        params: Dict[str, Any],
        base_spec: EngineSpec,
    ) -> SignalFrame:
        w = int(params.get("strategy.bb_window", base_spec.strategy.params.get("bb_window", 20)))
        k = float(params.get("strategy.bb_k", base_spec.strategy.params.get("bb_k", 2.0)))

        allow_short = bool(base_spec.strategy.params.get("allow_short", False))
        if "strategy.allow_short" in params:
            allow_short = bool(params["strategy.allow_short"])

        nan_policy = str(base_spec.strategy.params.get("nan_policy", "flat"))
        if "strategy.nan_policy" in params:
            nan_policy = str(params["strategy.nan_policy"])

        col_mid = f"sma_{w}"
        col_std = f"std_{w}"

        sig = pd.DataFrame(index=index, columns=symbols, dtype="float64")
        valid = pd.DataFrame(index=index, columns=symbols, dtype="bool")

        for sym in symbols:
            close = bars_close[sym]
            mid = bank[sym][col_mid]
            std = bank[sym][col_std]

            v = np.isfinite(close) & np.isfinite(mid) & np.isfinite(std)
            upper = mid + k * std
            lower = mid - k * std

            long_mask = close < lower
            if allow_short:
                short_mask = close > upper
                out = np.zeros(len(index), dtype=np.float64)
                out[long_mask] = 1.0
                out[short_mask] = -1.0
            else:
                out = long_mask.astype(np.float64)

            if nan_policy == "flat":
                out = np.where(v, out, 0.0)
            else:
                out = np.where(v, out, np.nan)

            sig[sym] = out
            valid[sym] = v

        return SignalFrame(
            signals=sig,
            validity=valid,
            meta={"adapter": "bollinger", "bb_window": w, "bb_k": k, "allow_short": allow_short, "nan_policy": nan_policy},
        )



STRATEGY_ADAPTERS: Dict[str, StrategyAdapter] = {
    "ma_cross": MACrossAdapter(),
    "sma_price": PriceAboveSMAAdapter(),
    "rsi": RSIStrategyAdapter(),
    "macd": MACDStrategyAdapter(),
    "bollinger": BollingerAdapter(),
}


# ============================================================
# Catalog helpers (optional but useful for Streamlit UI)
# ============================================================

def default_param_catalog(strategy_kind: str) -> Dict[str, ParamDef]:
    cat: Dict[str, ParamDef] = {}

    if strategy_kind == "ma_cross":
        cat["strategy.sma_fast_window"] = ParamDef("strategy.sma_fast_window", "int", (5, 60, 1), int)
        cat["strategy.sma_slow_window"] = ParamDef("strategy.sma_slow_window", "int", (20, 250, 1), int)
      

    elif strategy_kind == "sma_price":
        cat["strategy.sma_window"] = ParamDef("strategy.sma_window", "int", (10, 250, 1), int)
    
    elif strategy_kind == "rsi":
        cat["strategy.rsi_window"] = ParamDef("strategy.rsi_window", "int", (5, 100, 1), int)
        cat["strategy.rsi_oversold"] = ParamDef("strategy.rsi_oversold", "float", (10.0, 40.0, 10.0), float)
        cat["strategy.rsi_overbought"] = ParamDef("strategy.rsi_overbought", "float", (60.0, 90.0, 10.0), float)

    elif strategy_kind == "macd":
        cat["strategy.macd_fast_window"] = ParamDef("strategy.macd_fast_window", "int", (5, 50, 1), int)
        cat["strategy.macd_slow_window"] = ParamDef("strategy.macd_slow_window", "int", (20, 200, 1), int)
        cat["strategy.macd_signal_window"] = ParamDef("strategy.macd_signal_window", "int", (5, 50, 1), int)

    elif strategy_kind == "bollinger":
        cat["strategy.bb_window"] = ParamDef("strategy.bb_window", "int", (10, 100, 1), int)
        cat["strategy.bb_k"] = ParamDef("strategy.bb_k", "float", (2.0, 5.0, 0.5), float)
       
    # portfolio knobs you mentioned
    cat["portfolio.cooldown_bars"] = ParamDef("portfolio.cooldown_bars", "int", (0, 30, 1), int)
    cat["portfolio.buy_pct_cash"] = ParamDef("portfolio.buy_pct_cash", "float", (0.25, 1.1, 0.25), float)
    cat["portfolio.sell_pct_shares"] = ParamDef("portfolio.sell_pct_shares", "float", (0.25, 1.1, 0.25), float)

    return cat

def macd_pack_cached(
    close: np.ndarray,
    fast: int,
    slow: int,
    signal: int,
    ema_close_cache: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # cache EMA(close, span)
    ef = ema_close_cache.get(fast)
    if ef is None:
        ef = ema(close, fast)
        ema_close_cache[fast] = ef

    es = ema_close_cache.get(slow)
    if es is None:
        es = ema(close, slow)
        ema_close_cache[slow] = es

    macd_line = ef - es

    # cannot reuse across different (fast,slow), but cheap enough
    sig_line = ema(macd_line, signal)
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


# ============================================================
# Core optimization
# ============================================================

def run_optimization(
    base_spec: EngineSpec,
    active_params: List[ParamDef],
    cfg: OptimizeConfig,
) -> Tuple[TrialResult, pd.DataFrame, Dict[str, Any], EngineSpec, pd.DataFrame]:
    """
    Fast optimizer:
      - load MarketData once
      - precompute required features once (union of needed SMA windows)
      - per trial: generate signals from precomputed arrays (adapter), apply cooldown via PortfolioConfig, run portfolio stats fast

    Returns:
      best_result, top_df, best_params, best_spec
    """
    if not active_params:
        raise ValueError("active_params is empty; nothing to optimize.")

    strategy_kind = base_spec.strategy.kind.lower()
    if strategy_kind not in STRATEGY_ADAPTERS:
        raise ValueError(f"No StrategyAdapter registered for strategy kind '{strategy_kind}'")

    adapter = STRATEGY_ADAPTERS[strategy_kind]

    # 1) Load data once (respect base_spec.data start/end to define the available universe)
    md_full = _load_market_data_from_spec(base_spec.data)

    symbols = list(base_spec.data.symbols)
    if not symbols:
        raise ValueError("No symbols in base_spec.data.symbols")

    # 2) Build a common index once (inner intersection across symbols for robustness)
    #    This ensures arrays are aligned and portfolio doesn't hit missing timestamps.
    md = _align_marketdata_inner(md_full, symbols)

    common_index = md.bars[symbols[0]].index
    if len(common_index) < 2:
        raise ValueError("Not enough bars after alignment; need at least 2 timestamps.")


    # --- aligned price arrays ONCE ---
    bars_open: Dict[str, np.ndarray] = {}
    bars_close: Dict[str, np.ndarray] = {}
    bars_vol: Dict[str, np.ndarray] = {}
    need_volume = bool(base_spec.portfolio.use_participation_cap or base_spec.portfolio.use_volume_gate)
    adv_by_window_np: Dict[int, Dict[str, np.ndarray]] = {}

    for s in symbols:
        b = md.bars[s].reindex(common_index)  # aligned already; reindex ok & explicit
        bars_open[s] = b["Open"].to_numpy(dtype=np.float64, copy=False)
        bars_close[s] = b["Close"].to_numpy(dtype=np.float64, copy=False)
        if need_volume:
            vcol = base_spec.portfolio.volume_col
            if vcol not in b.columns:
                raise KeyError(f"Missing volume column '{vcol}' for {s}. Available: {list(b.columns)}")
            bars_vol[s] = b[vcol].to_numpy(dtype=np.float64, copy=False)
            
    # 3) Precompute arrays once (ALWAYS) + SMA bank (IF NEEDED)
    req = adapter.required_bank(base_spec, active_params)
    bank = build_bank(bars_close, req)
        
    if need_volume:
        adv_windows: List[int] = []
        if base_spec.portfolio.use_participation_cap and str(base_spec.portfolio.participation_basis) == "adv":
            adv_windows.append(int(base_spec.portfolio.adv_window))
        if base_spec.portfolio.use_volume_gate and str(base_spec.portfolio.volume_gate_kind) == "min_ratio_adv":
            adv_windows.append(int(base_spec.portfolio.volume_gate_adv_window))
        adv_windows = sorted(set(w for w in adv_windows if w >= 1))

        for w in adv_windows:
            adv_by_window_np[w] = {}
            for s in symbols:
                v = bars_vol[s]
                # min_periods=1 behavior
                c = np.cumsum(np.insert(v, 0, 0.0))
                adv = np.empty_like(v)
                # first part: min_periods=1
                idx = np.arange(v.size)
                lo = np.maximum(0, idx - (w - 1))
                den = (idx - lo + 1).astype(np.float64)
                adv[:] = (c[idx + 1] - c[lo]) / den
                adv_by_window_np[w][s] = adv

    




    # 4) Candidate iterator
    method = cfg.method.lower()
    if method == "grid":
        candidates = _iter_grid([p for p in active_params if p.enabled])
    elif method == "random":
        candidates = _iter_random([p for p in active_params if p.enabled], n_trials=int(cfg.n_trials), seed=int(cfg.seed))
    else:
        raise ValueError(f"Unknown optimization method: {cfg.method}")

    # 5) Evaluate
    results: List[TrialResult] = []
    for params in candidates:
        ok, err = adapter.validate_params(params, base_spec)
        if not ok:
            results.append(TrialResult(
                params=params,
                pnl=float("-inf"),
                traded_notional=0.0,
                efficiency=float("-inf"),
                n_fills=0,
                cagr=float("-inf"),
                error=err or "invalid params",
            ))
            continue

        r = _eval_one_trial(
            base_spec=base_spec,
            md=md,
            common_index=common_index,
            bank=bank,
            bars_open=bars_open,      # NEW
            bars_close=bars_close,
            bars_vol=bars_vol,
            adv_by_window_np=adv_by_window_np,
            adapter=adapter,
            params=params,
        )

        results.append(r)

    df = pd.DataFrame([{
        **r.params,
        "pnl": r.pnl,
        "traded_notional": r.traded_notional,
        "efficiency": r.efficiency,
        "n_fills": r.n_fills,
        "cagr": r.cagr,
        "error": r.error,
    } for r in results])

    # rank valid rows by (pnl desc, efficiency desc)
    df_valid = df[df["error"].isna()].copy()
    if df_valid.empty:
        ranked_df = df.sort_values(["pnl", "cagr"], ascending=[False, False]).reset_index(drop=True)
        top_df = ranked_df.head(int(cfg.top_k)).reset_index(drop=True)

        # ---- add last-bar ("today") signal columns to leaderboard ----
        sig_dates: List[pd.Timestamp] = []
        sig_nums: List[float] = []
        sig_labels: List[str] = []

        for _, row in top_df.iterrows():
            row_params = _extract_params_from_row(row.to_dict())
            d, n, lab = _latest_signal_for_params(
                adapter=adapter,
                base_spec=base_spec,
                symbols=symbols,
                index=common_index,
                bank=bank,
                bars_close=bars_close,
                params=row_params,
            )
            sig_dates.append(d)
            sig_nums.append(n)
            sig_labels.append(lab)

        top_df["signal_date"] = sig_dates
        top_df["signal_today"] = sig_nums
        top_df["signal_label"] = sig_labels


        best_row = top_df.iloc[0].to_dict()
        best_params = {k: best_row[k] for k in best_row.keys() if k not in ("pnl", "traded_notional", "cagr", "n_fills", "error")}
        best = TrialResult(
            params=best_params,
            pnl=float(best_row["pnl"]),
            traded_notional=float(best_row.get("traded_notional", 0.0)),
            efficiency=float(best_row.get("efficiency", float("-inf"))),
            n_fills=int(best_row.get("n_fills", 0)),
            cagr=float(best_row["cagr"]),
            error=str(best_row.get("error")) if best_row.get("error") is not None else None,
        )
        best_spec = _apply_params_to_spec(base_spec, best.params)
        return best, top_df, best_params, best_spec, ranked_df

    ranked_df = df_valid.sort_values(["pnl", "cagr"], ascending=[False, False]).reset_index(drop=True)
    df_valid = df_valid.sort_values(["pnl", "cagr"], ascending=[False, False])
    top_df = ranked_df.head(int(cfg.top_k)).reset_index(drop=True)
    # ---- add last-bar ("today") signal columns to leaderboard ----
    sig_dates: List[pd.Timestamp] = []
    sig_nums: List[float] = []
    sig_labels: List[str] = []

    for _, row in top_df.iterrows():
        row_params = _extract_params_from_row(row.to_dict())
        d, n, lab = _latest_signal_for_params(
            adapter=adapter,
            base_spec=base_spec,
            symbols=symbols,
            index=common_index,
            bank=bank,
            bars_close=bars_close,
            params=row_params,
        )
        sig_dates.append(d)
        sig_nums.append(n)
        sig_labels.append(lab)

    top_df["signal_date"] = sig_dates
    top_df["signal_today"] = sig_nums
    top_df["signal_label"] = sig_labels

    

    best_row = top_df.iloc[0].to_dict()
    best_params = {k: best_row[k] for k in best_row.keys() if k not in ("pnl", "traded_notional", "cagr", "n_fills", "error")}
    best = TrialResult(
        params=best_params,
        pnl=float(best_row["pnl"]),
        traded_notional=float(best_row["traded_notional"]),
        efficiency=float(best_row["efficiency"]),
        n_fills=int(best_row["n_fills"]),
        cagr=float(best_row["cagr"]),
        error=None,
    )
    best_spec = _apply_params_to_spec(base_spec, best.params)
    return best, top_df, best_params, best_spec, ranked_df

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
def eval_stats_only_for_spec_arrays(
    spec: EngineSpec,
) -> Dict[str, Any]:
    """
    Runs one fast stats-only backtest for a spec and returns a flat dict.
    Uses same alignment logic + PortfolioEngine.run_stats_only_arrays.
    """
    # Load + align
    md_full = _load_market_data_from_spec()
    symbols = list(spec.data.symbols)
    md = _align_marketdata_inner(md_full, symbols)

    if len(symbols) != 1:
        raise ValueError("stats_only eval currently supports single-symbol (same as optimize fast path).")

    sym = symbols[0]
    idx = md.bars[sym].index
    if len(idx) < 2:
        raise ValueError("Not enough bars after alignment.")
    
    # Apply data window from spec.data.start/end (the same semantics as optimization)
    idx2 = _slice_index(idx, spec.data.start, spec.data.end)
    if idx2 is None or len(idx2) < 2:
        raise ValueError("Window too small after slicing.")
    b = md.bars[sym].loc[idx2[0]:idx2[-1]]

    open_px  = b["Open"].to_numpy(dtype=np.float64, copy=False)
    close_px = b["Close"].to_numpy(dtype=np.float64, copy=False)

    # NEW: volume/ADV support (so volume gate & participation cap work)
    need_volume = bool(spec.portfolio.use_participation_cap or spec.portfolio.use_volume_gate)

    vol_px = None
    adv_cap_px = None
    adv_gate_px = None

    if need_volume:
        vcol = spec.portfolio.volume_col
        if vcol not in b.columns:
            raise KeyError(f"Missing volume column '{vcol}' in bars for {sym}. Available: {list(b.columns)}")
        vol_px = b[vcol].to_numpy(dtype=np.float64, copy=False)

        # Build ADV arrays only for the windows actually needed
        adv_windows = []
        if spec.portfolio.use_participation_cap and str(spec.portfolio.participation_basis) == "adv":
            adv_windows.append(int(spec.portfolio.adv_window))
        if spec.portfolio.use_volume_gate and str(spec.portfolio.volume_gate_kind) == "min_ratio_adv":
            adv_windows.append(int(spec.portfolio.volume_gate_adv_window))
        adv_windows = sorted(set(w for w in adv_windows if w >= 1))

        adv_by_w = {}
        for w in adv_windows:
            v = vol_px
            c = np.cumsum(np.insert(v, 0, 0.0))
            adv = np.empty_like(v)
            idx = np.arange(v.size)
            lo = np.maximum(0, idx - (w - 1))
            den = (idx - lo + 1).astype(np.float64)
            adv[:] = (c[idx + 1] - c[lo]) / den
            adv_by_w[w] = adv

        if spec.portfolio.use_participation_cap and str(spec.portfolio.participation_basis) == "adv":
            adv_cap_px = adv_by_w[int(spec.portfolio.adv_window)]

        if spec.portfolio.use_volume_gate and str(spec.portfolio.volume_gate_kind) == "min_ratio_adv":
            adv_gate_px = adv_by_w[int(spec.portfolio.volume_gate_adv_window)]


    # Build signals using strategy adapter + SMA bank (minimal compute: only needed windows)
    sk = spec.strategy.kind.lower()
    if sk not in STRATEGY_ADAPTERS:
        raise ValueError(f"No adapter for strategy kind '{sk}'")

    adapter = STRATEGY_ADAPTERS[sk]

    # Build SMA bank only for required windows of THIS spec (not full domain)
    # We can reuse your sma builder
    def _sma_bank_numpy(close: np.ndarray, windows: List[int]) -> Dict[str, np.ndarray]:
        n = close.size
        csum = np.cumsum(np.insert(close.astype(np.float64, copy=False), 0, 0.0))
        out: Dict[str, np.ndarray] = {}
        for w0 in windows:
            w = int(w0)
            sma = np.full(n, np.nan, dtype=np.float64)
            if 0 < w <= n:
                sma[w - 1 :] = (csum[w:] - csum[:-w]) / w
            out[f"sma_{w}"] = sma
        return out

    # Determine required SMA windows from current spec params only
    if sk == "sma_price":
        w = int(spec.strategy.params.get("window", 50))
        req = [w]
    elif sk == "ma_cross":
        f = int(spec.strategy.params.get("fast_window", 15))
        s = int(spec.strategy.params.get("slow_window", 50))
        req = [f, s]
    elif sk == "rsi":
        w = int(spec.strategy.params.get("rsi_window", 14))
        os= int(spec.strategy.params.get("rsi_oversold", 30))
        ob= int(spec.strategy.params.get("rsi_overbought", 70))
        req = [w,ob,os]
    elif sk == "macd":
        f = int(spec.strategy.params.get("macd_fast_window", 12))
        s = int(spec.strategy.params.get("macd_slow_window", 26))
        sig = int(spec.strategy.params.get("macd_signal_window", 9))
        req = [f, s, sig]
    elif sk == "bollinger":
        w = int(spec.strategy.params.get("bb_window", 20))
        k = float(spec.strategy.params.get("bb_k", 2.0))
        req = [w]

    bars_close = {sym: close_px}
    req = adapter.required_bank(spec, active_params=[])  # spec-only: no domain
    bank = build_bank(bars_close, req)

    sf = adapter.make_signals_from_bank(
        symbols=[sym],
        index=b.index,   # already sliced window
        bank=bank,
        bars_close=bars_close,
        params={},       # no overrides, spec.params only
        base_spec=spec,
    )
    sig = sf.signals[sym].to_numpy(dtype=np.float64, copy=False)

    
    # Portfolio stats only
    port = PortfolioEngine(spec.portfolio)
    stats = port.run_stats_only_arrays(open_px=open_px, close_px=close_px, sig=sig, vol_px=vol_px, adv_cap_px=adv_cap_px, adv_gate_px=adv_gate_px)
    E0 = float(spec.portfolio.initial_cash)
    pnl = float(stats.pnl)

    # prefer final_equity if available, else infer
    ET = float(getattr(stats, "final_equity", E0 + pnl))

    n = int(close_px.size)
    ppy = float(getattr(spec, "periods_per_year", 252))

    if n <= 1 or E0 <= 0 or ET <= 0:
        cagr = np.nan
    else:
        cagr = (ET / E0) ** (ppy / n) - 1.0

    # Flatten into dict for UI table
    return {
        "pnl": float(stats.pnl),
        "traded_notional": float(stats.traded_notional),
        "n_fills": int(stats.n_fills),
        "cagr": float(cagr) if np.isfinite(cagr) else np.nan,
        # keep your efficiency logic if stats has volume_inv; else use traded_notional
        "efficiency": 1.0 if float(getattr(stats, "volume_inv", stats.traded_notional)) <= 0 else float(stats.pnl) / float(getattr(stats, "volume_inv", stats.traded_notional)),
        # optional extras if available
        "final_equity": float(getattr(stats, "final_equity", np.nan)),
        "max_drawdown": float(getattr(stats, "max_drawdown", np.nan)),
    }

from datetime import timedelta

def batch_optimize_by_period(
    base_spec: EngineSpec,
    active_params: List[ParamDef],
    cfg: OptimizeConfig,
    periods: Dict[str, Tuple[str, str]],
    selected_period_labels: List[str],
    objective: str = "pnl",
) -> pd.DataFrame:

    rows = []

    for label in selected_period_labels:
        p_start, p_end = periods[label]

        per_spec = replace(
            base_spec,
            data=replace(base_spec.data, start=str(p_start), end=str(p_end)),
        )

        best, top_df, best_params, best_spec, ranked_df = run_optimization(
            base_spec=per_spec,
            active_params=active_params,
            cfg=cfg,
        )

        # ---- stats-only evaluation (fast) ----
        stats = eval_stats_only_for_spec_arrays(best_spec)
        score = stats.get(objective, np.nan)

        # ✅ CREATE row FIRST
        row = {
            "period": label,
            "start": p_start,
            "end": p_end,
            "objective": objective,
            "objective_value": float(score) if score is not None else np.nan,
        }

        # record best params (flatten)
        for k, v in best_params.items():
            row[f"param.{k}"] = v

        # record stats-only (flatten)
        for k, v in stats.items():
            row[f"stat.{k}"] = v

        # ---- FULL run ONLY for best spec (fills -> trade ledger/perf) ----
        try:
            best_bundle = BacktestEngine(best_spec).run()

            trades_df = best_bundle.report.tables.get("trades", pd.DataFrame())
            ledger = best_bundle.report.tables.get("trade_ledger", pd.DataFrame())
            tperf  = best_bundle.report.tables.get("trade_performance", pd.DataFrame())

            row["best.trades"] = int(len(trades_df))
            row["best.ledger_trades"] = int(len(ledger))

            if (tperf is not None) and (not tperf.empty) and ("Value" in tperf.columns) and ("Win Rate" in tperf.index):
                row["best.win_rate"] = float(tperf.loc["Win Rate", "Value"])
            else:
                row["best.win_rate"] = np.nan

        except Exception as e:
            # Don't kill the batch if full run fails; keep audit trail
            row["best.trades"] = np.nan
            row["best.ledger_trades"] = np.nan
            row["best.win_rate"] = np.nan
            row["best.full_run_error"] = f"{type(e).__name__}: {e}"

        rows.append(row)

    return pd.DataFrame(rows)



    

def build_spec_from_result_row(base_spec: EngineSpec, row: Any) -> EngineSpec:
    """
    Accepts either:
      - optimizer-ranked rows with raw keys: "strategy.sma_window", "portfolio.cooldown_bars", ...
      - batch rows with prefixed keys: "param.strategy.sma_window", "param.portfolio.cooldown_bars", ...

    Ignores:
      - metrics: pnl, cagr, n_fills, traded_notional, efficiency, error
      - batch/meta columns: period, start, end, objective, objective_value
      - stat.* columns
    """
    if isinstance(row, pd.Series):
        d = row.to_dict()
    else:
        d = dict(row)

    metric_cols = {"pnl", "traded_notional", "cagr", "n_fills", "efficiency", "error"}
    meta_cols = {"period", "start", "end", "objective", "objective_value"}

    params: Dict[str, Any] = {}

    for k, v in d.items():
        if k in metric_cols or k in meta_cols:
            continue
        if isinstance(k, str) and k.startswith("stat."):
            continue

        # batch format: param.<real_key>
        if isinstance(k, str) and k.startswith("param."):
            real_k = k[len("param."):]
            params[real_k] = v
            continue

        # normal optimization format: real key already
        if isinstance(k, str) and (k.startswith("strategy.") or k.startswith("portfolio.") or k == "data.window"):
            params[k] = v

    return _apply_params_to_spec(base_spec, params)

def _eval_one_trial_slow_pandas(*args, **kwargs) -> TrialResult:
    """Fallback path for multi-symbol trials.

    Not implemented in this project snapshot. If you hit this, either:
      - restrict optimization to a single symbol, or
      - implement a multi-symbol portfolio simulation + stats extraction.
    """
    return TrialResult(params=kwargs.get("params", {}), pnl=float("-inf"), traded_notional=0.0, efficiency=float("-inf"), n_fills=0,cagr=float("-inf"), error="multi-symbol optimize not implemented")

# ============================================================
# Trial evaluation
# ============================================================
def _eval_one_trial(
    base_spec: EngineSpec,
    md: MarketData,
    common_index: pd.DatetimeIndex,
    bank: Dict[str, Dict[str, np.ndarray]],
    bars_open: Dict[str, np.ndarray],
    bars_close: Dict[str, np.ndarray],
    bars_vol: Dict[str, np.ndarray],
    adv_by_window_np: Dict[int, Dict[str, np.ndarray]],
    adapter: StrategyAdapter,
    params: Dict[str, Any],
) -> TrialResult:
    try:
        symbols = list(base_spec.data.symbols)
        if len(symbols) != 1:
            # Fallback to your old logic (multi-symbol) for now
            return _eval_one_trial_slow_pandas(
                base_spec=base_spec,
                md=md,
                common_index=common_index,
                bank=bank,
                bars_close=bars_close,
                adapter=adapter,
                params=params,
            )

        sym = symbols[0]

        # ----------------------------
        # A) Select the backtest window (optional)
        # ----------------------------
        # If you have a param like params["data.window"] = ("2020-01-01","2022-12-31")
        # Otherwise just use the full aligned arrays.
        idx_pos = None
        if "data.window" in params and params["data.window"] is not None:
            start, end = params["data.window"]
            idx_slice = common_index
            if start is not None:
                idx_slice = idx_slice[idx_slice >= pd.to_datetime(start)]
            if end is not None:
                idx_slice = idx_slice[idx_slice <= pd.to_datetime(end)]
            if len(idx_slice) < 2:
                return TrialResult(params=params, pnl=float("-inf"), traded_notional=0.0, efficiency=float("-inf"), n_fills=0,cagr=float("-inf"), error="window too small")
            idx_pos = common_index.get_indexer_for(idx_slice)

        open_px = bars_open[sym] if idx_pos is None else bars_open[sym][idx_pos]
        close_px = bars_close[sym] if idx_pos is None else bars_close[sym][idx_pos]
        need_volume = bool(base_spec.portfolio.use_participation_cap or base_spec.portfolio.use_volume_gate)
        vol_px = (bars_vol[sym] if idx_pos is None else bars_vol[sym][idx_pos]) if need_volume else None
        adv_cap_px = None
        adv_gate_px = None

        port_base = base_spec.portfolio

        if port_base.use_participation_cap and str(port_base.participation_basis) == "adv":
            w = int(port_base.adv_window)
            a = adv_by_window_np.get(w, {}).get(sym)
            if a is None:
                return TrialResult(params=params, pnl=float("-inf"), traded_notional=0.0, efficiency=float("-inf"), n_fills=0, cagr=float("-inf"),
                                error=f"missing ADV cap series for window={w}")
            adv_cap_px = a if idx_pos is None else a[idx_pos]

        if port_base.use_volume_gate and str(port_base.volume_gate_kind) == "min_ratio_adv":
            w = int(port_base.volume_gate_adv_window)
            a = adv_by_window_np.get(w, {}).get(sym)
            if a is None:
                return TrialResult(params=params, pnl=float("-inf"), traded_notional=0.0, efficiency=float("-inf"), n_fills=0,cagr=float("-inf"),
                                error=f"missing ADV gate series for window={w}")
            adv_gate_px = a if idx_pos is None else a[idx_pos]




        # ----------------------------
        # B) Build signals using adapter (single source of truth)
        # ----------------------------
        bars_close_trial = {sym: close_px}

        # If you slice idx_pos, the precomputed bank arrays must also be sliced.
        # Easiest: build a "view" bank for this sym using slicing.
        sym_bank = bank[sym]
        if idx_pos is not None:
            sym_bank = {k: v[idx_pos] for k, v in sym_bank.items()}

        sf = adapter.make_signals_from_bank(
            symbols=[sym],
            index=(common_index if idx_pos is None else common_index[idx_pos]),
            bank={sym: sym_bank},
            bars_close=bars_close_trial,
            params=params,
            base_spec=base_spec,
        )
        sig = sf.signals[sym].to_numpy(dtype=np.float64, copy=False)

        # ----------------------------
        # C) Build portfolio config for this trial (apply params)
        # ----------------------------
        port_cfg = _apply_portfolio_params(base_spec.portfolio, params)
        port = PortfolioEngine(port_cfg)

        # IMPORTANT: call your NEW arrays fast path
        stats = port.run_stats_only_arrays(open_px=open_px, close_px=close_px, sig=sig, vol_px=vol_px, adv_cap_px=adv_cap_px, adv_gate_px=adv_gate_px)

        pnl = float(stats.pnl)

        E0 = float(base_spec.portfolio.initial_cash)

        # Prefer final_equity if PortfolioEngine exposes it; else infer from pnl
        ET = float(getattr(stats, "final_equity", E0 + pnl))

        n = int(close_px.size)
        ppy = float(getattr(base_spec, "periods_per_year", 252))

        # CAGR = (ET/E0)^(ppy/n) - 1
        if n <= 1 or E0 <= 0 or ET <= 0:
            cagr = float("-inf")
        else:
            cagr = (ET / E0) ** (ppy / n) - 1.0


        traded = float(stats.traded_notional)
        n_fills = int(stats.n_fills)
        # Efficiency (performance) requested:
        #   if volume_inv <= 0: efficiency = 1.0 (100%)
        #   else: efficiency = pnl / volume_inv
        # NOTE: To fully implement the *monthly reset at last SELL*, the portfolio stats layer
        # must expose per-fill cashflows with timestamps. If unavailable, we fall back to using
        # traded_notional as a proxy denominator (keeps optimizer runnable).
        volume_inv = (
            getattr(stats, "volume_inv", None)
            or getattr(stats, "volume_invested", None)
            or getattr(stats, "volumeinv", None)
        )
        if volume_inv is None:
            volume_inv = traded
        volume_inv = float(volume_inv)
        eff = 1.0 if volume_inv <= 0 else float(pnl / volume_inv)

        return TrialResult(
            params=params,
            pnl=pnl,
            traded_notional=traded,
            efficiency=eff,
            n_fills=n_fills,
            error=None,
            cagr=float(cagr),
        )

    except Exception as e:
        return TrialResult(
            params=params,
            pnl=float("-inf"),
            traded_notional=0.0,
            efficiency=float("-inf"),
            n_fills=0,
            cagr=float("-inf"),
            error=str(e),
        )




# ============================================================
# Candidate generation
# ============================================================

def _expand_domain(p: ParamDef) -> List[Any]:
    if p.kind in ("choice", "date_window"):
        return list(p.domain)
    if p.kind == "int":
        lo, hi, step = p.domain
        return list(range(int(lo), int(hi) + 1, int(step)))
    if p.kind == "float":
        lo, hi, step = map(float, p.domain)
        n = int(math.floor((hi - lo) / step)) + 1
        return [lo + i * step for i in range(n)]
    raise ValueError(f"Unknown ParamDef kind: {p.kind}")

def _iter_grid(active: List[ParamDef]) -> Iterable[Dict[str, Any]]:
    grids = [_expand_domain(p) for p in active]
    keys = [p.key for p in active]
    casts = [p.cast for p in active]
    for combo in itertools.product(*grids):
        out: Dict[str, Any] = {}
        for k, v, c in zip(keys, combo, casts):
            out[k] = c(v)
        yield out

def _iter_random(active: List[ParamDef], n_trials: int, seed: int) -> Iterable[Dict[str, Any]]:
    rng = random.Random(seed)
    grids = {p.key: _expand_domain(p) for p in active}
    casts = {p.key: p.cast for p in active}
    keys = [p.key for p in active]
    for _ in range(n_trials):
        out: Dict[str, Any] = {}
        for k in keys:
            out[k] = casts[k](rng.choice(grids[k]))
        yield out


def _domain_values_int(active: List[ParamDef], key: str) -> List[int]:
    p = next((x for x in active if x.key == key and x.enabled), None)
    if p is None:
        return []

    # Accept manual-list domains too (UI converts int->choice)
    if p.kind not in ("int", "choice"):
        return []

    vals = _expand_domain(p)
    out = []
    for v in vals:
        try:
            out.append(int(v))
        except Exception:
            pass
    return sorted(set(out))



# ============================================================
# Apply params to spec/configs (for returning best_spec)
# ============================================================

def _apply_params_to_spec(base_spec: EngineSpec, params: Dict[str, Any]) -> EngineSpec:
    # DataConfig: apply date window if optimized
    data_cfg = base_spec.data
    if "data.window" in params and params["data.window"] is not None:
        start, end = params["data.window"]
        data_cfg = replace(data_cfg, start=str(start), end=str(end))

    # StrategyConfig: apply strategy params
    strat_cfg = base_spec.strategy
    sp = dict(strat_cfg.params or {})
    if "strategy.sma_fast_window" in params:
        sp["fast_window"] = int(params["strategy.sma_fast_window"])
    if "strategy.sma_slow_window" in params:
        sp["slow_window"] = int(params["strategy.sma_slow_window"])
    if "strategy.sma_window" in params:
        sp["window"] = int(params["strategy.sma_window"])
    if "strategy.bb_window" in params:
        sp["bb_window"] = int(params["strategy.bb_window"])
    if "strategy.bb_k" in params:
        sp["bb_k"] = float(params["strategy.bb_k"])
    if "strategy.rsi_window" in params:
        sp["rsi_window"] = int(params["strategy.rsi_window"])
    if "strategy.rsi_oversold" in params:
        sp["rsi_oversold"] = float(params["strategy.rsi_oversold"])
    if "strategy.rsi_overbought" in params:
        sp["rsi_overbought"] = float(params["strategy.rsi_overbought"])
    if "strategy.macd_fast_window" in params:
        sp["macd_fast_window"] = int(params["strategy.macd_fast_window"])
    if "strategy.macd_slow_window" in params:   
        sp["macd_slow_window"] = int(params["strategy.macd_slow_window"])
    if "strategy.macd_signal_window" in params:
        sp["macd_signal_window"] = int(params["strategy.macd_signal_window"])
    if "strategy.allow_short" in params:
        sp["allow_short"] = bool(params["strategy.allow_short"])
    if "strategy.nan_policy" in params:
        sp["nan_policy"] = str(params["strategy.nan_policy"])
    strat_cfg = replace(strat_cfg, params=sp)

    # PortfolioConfig
    port_cfg = _apply_portfolio_params(base_spec.portfolio, params)

    return replace(base_spec, data=data_cfg, strategy=strat_cfg, portfolio=port_cfg)


def _apply_portfolio_params(port_cfg: PortfolioConfig, params: Dict[str, Any]) -> PortfolioConfig:
    upd = port_cfg
    if "portfolio.cooldown_bars" in params:
        upd = replace(upd, cooldown_bars=int(params["portfolio.cooldown_bars"]))
    if "portfolio.buy_pct_cash" in params:
        upd = replace(upd, buy_pct_cash=float(params["portfolio.buy_pct_cash"]))
    if "portfolio.sell_pct_shares" in params:
        upd = replace(upd, sell_pct_shares=float(params["portfolio.sell_pct_shares"]))
    return upd


# ============================================================
# Data loading + alignment + slicing
# ============================================================

def _load_market_data_from_spec(cfg: DataConfig) -> MarketData:
    if cfg.source == "bmce":
        if cfg.bmce_paths is None:
            raise ValueError("BMCE source selected but bmce_paths is None.")
        ds = BMCEDataSource(timezone=cfg.timezone)
        return ds.load(
            symbols=cfg.symbols,
            start=cfg.start,
            end=cfg.end,
            interval=cfg.interval,
            paths=cfg.bmce_paths,
        )

    if cfg.source == "yfinance":
        ds = YahooFinanceDataSource(timezone=cfg.timezone)
        # Note: your YahooFinanceDataSource.load supports start/end/interval + kwargs
        return ds.load(
            symbols=cfg.symbols,
            start=cfg.start,
            end=cfg.end,
            interval=cfg.interval,
            auto_adjust=cfg.yf_auto_adjust,
            progress=False,
        )

    raise ValueError(f"Unknown data source: {cfg.source}")


def _align_marketdata_inner(md: MarketData, symbols: List[str]) -> MarketData:
    # inner intersection index across symbols (robust multi-asset)
    idx = md.bars[symbols[0]].index
    for s in symbols[1:]:
        idx = idx.intersection(md.bars[s].index)
    idx = idx.sort_values()

    bars_aligned: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        bars_aligned[s] = md.bars[s].reindex(idx)

    return MarketData(
        bars=bars_aligned,
        source=md.source,
        timezone=md.timezone,
        interval=md.interval,
        meta=dict(md.meta),
    )


def _slice_marketdata(md: MarketData, symbols: List[str], index: pd.DatetimeIndex) -> MarketData:
    """Slice MarketData to a given DatetimeIndex.

    Optimization hot loop calls this only when you optimize data.window.
    Prefer a cheap .loc[start:end] slice when possible; fall back to reindex
    only if the slice doesn't exactly match the requested index.
    """
    bars: Dict[str, pd.DataFrame] = {}

    if len(index) == 0:
        for s in symbols:
            bars[s] = md.bars[s].iloc[0:0].reindex(index)
        return MarketData(
            bars=bars,
            source=md.source,
            timezone=md.timezone,
            interval=md.interval,
            meta=dict(md.meta),
        )

    start = index[0]
    end = index[-1]

    for s in symbols:
        df = md.bars[s]
        df_span = df.loc[start:end]
        # If df_span already has the exact index, keep it; else reindex
        if df_span.index.equals(index):
            bars[s] = df_span
        else:
            bars[s] = df.reindex(index)

    return MarketData(
        bars=bars,
        source=md.source,
        timezone=md.timezone,
        interval=md.interval,
        meta=dict(md.meta),
    )

def _slice_index(index: pd.DatetimeIndex, start: Optional[str], end: Optional[str]) -> Optional[pd.DatetimeIndex]:
    if start is None and end is None:
        return index
    tz = index.tz
    s = pd.Timestamp(start, tz=tz) if start is not None else None
    e = pd.Timestamp(end, tz=tz) if end is not None else None
    out = index
    if s is not None:
        out = out[out >= s]
    if e is not None:
        out = out[out <= e]
    return out if len(out) > 0 else None


