# indicators.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
import hashlib
import json
import os
import pickle

import pandas as pd
import numpy as np

# ---------- Registry ----------

IndicatorFunc = Callable[[pd.DataFrame, Dict[str, Any]], Union[pd.Series, pd.DataFrame]]

_INDICATOR_REGISTRY: Dict[str, IndicatorFunc] = {}

def register_indicator(name: str) -> Callable[[IndicatorFunc], IndicatorFunc]:
    def deco(fn: IndicatorFunc) -> IndicatorFunc:
        if name in _INDICATOR_REGISTRY:
            raise ValueError(f"Indicator '{name}' already registered")
        _INDICATOR_REGISTRY[name] = fn
        return fn
    return deco


# ---------- Specs & Containers ----------

@dataclass(frozen=True)
class FeatureSpec:
    """
    Declarative spec of a feature/indicator.
    """
    indicator: str                      # registry key, e.g. "sma"
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: Tuple[str, ...] = ("Close",)  # required columns from bars
    name: Optional[str] = None          # optional override for base name
    warmup: Optional[int] = None        # bars to treat as invalid (NaN)
    output_mode: str = "auto"           # "auto" | "series" | "df"

    # indicators.py (inside FeatureSpec)

    # inside FeatureSpec
    def canonical_name(self) -> str:
        """
        Naming contract:
        - if spec.name is provided, it is authoritative (do NOT append params again)
        - otherwise, auto-generate a stable name from indicator + key params
        """
        if self.name:  # authoritative override
            return self.name

        # fallback auto-name (keep stable)
        p = self.params or {}

        # common patterns
        if self.indicator == "sma":
            w = p.get("window", p.get("period", p.get("n")))
            return f"sma_{w}" if w is not None else "sma"
        
        if self.indicator == "rsi":
            n = p.get("period", p.get("window", p.get("n")))
            return f"rsi_{n}" if n is not None else "rsi"

        if self.indicator == "macd":
            fast = p.get("fast", 12)
            slow = p.get("slow", 26)
            sig  = p.get("signal", 9)
            return f"macd_{fast}_{slow}_{sig}"
        
        if self.indicator in ("rolling_std", "std", "stdev", "vol"):
            w = p.get("window", p.get("period", p.get("n")))
            return f"std_{w}" if w is not None else "std"

        # generic fallback
        # you can expand this per-indicator later for nicer naming
        if p:
            # stable, sorted param encoding
            suffix = "_".join(f"{k}={p[k]}" for k in sorted(p.keys()))
            return f"{self.indicator}_{suffix}"
        return self.indicator



    def spec_hash(self, version: str = "v1") -> str:
        payload = {
            "indicator": self.indicator,
            "params": {k: self.params[k] for k in sorted(self.params)},
            "inputs": list(self.inputs),
            "name": self.name,
            "warmup": self.warmup,
            "output_mode": self.output_mode,
            "version": version,
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass
class FeaturesData:
    """
    Mirror of MarketData but for computed features.
    """
    features: Dict[str, pd.DataFrame]
    source: str
    timezone: str
    interval: str
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------- Engine ----------

class IndicatorEngine:
    """
    Orchestrates feature computation with validation, naming, warmup policy, and caching.
    """
    def __init__(
        self,
        cache_dir: Optional[str] = ".cache/features",
        enable_disk_cache: bool = True,
        enable_memory_cache: bool = True,
        engine_version: str = "v1",
    ) -> None:
        self.cache_dir = cache_dir
        self.enable_disk_cache = enable_disk_cache
        self.enable_memory_cache = enable_memory_cache
        self.engine_version = engine_version
        self._mem_cache: Dict[str, pd.DataFrame] = {}
        self._framehash_cache: Dict[Tuple[int, Tuple[str, ...]], str] = {}

        if self.enable_disk_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def compute(
        self,
        market_data,                       # expects your MarketData container
        specs: List[FeatureSpec],
        symbols: Optional[List[str]] = None,
    ) -> FeaturesData:
        bars_dict: Dict[str, pd.DataFrame] = market_data.bars
        symbols = symbols or list(bars_dict.keys())

        out: Dict[str, pd.DataFrame] = {}
        run_meta: Dict[str, Any] = {"specs": [], "engine_version": self.engine_version}

        for sym in symbols:
            bars = bars_dict[sym]
            feats_sym = []

            for spec in specs:
                df_feat, meta = self._compute_one_symbol_one_spec(sym, bars, spec)
                feats_sym.append(df_feat)
                run_meta["specs"].append(meta)

            # column-wise concat with shared index
            out[sym] = pd.concat(feats_sym, axis=1).sort_index()

        return FeaturesData(
            features=out,
            source=getattr(market_data, "source", "unknown"),
            timezone=getattr(market_data, "timezone", "unknown"),
            interval=getattr(market_data, "interval", "unknown"),
            meta=run_meta,
        )

    def _compute_one_symbol_one_spec(
        self,
        symbol: str,
        bars: pd.DataFrame,
        spec: FeatureSpec
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:

        if spec.indicator not in _INDICATOR_REGISTRY:
            raise KeyError(f"Unknown indicator '{spec.indicator}'. Registered: {list(_INDICATOR_REGISTRY)}")

        # Validate required inputs
        missing = [c for c in spec.inputs if c not in bars.columns]
        if missing:
            raise ValueError(f"{symbol}: missing required columns for {spec.indicator}: {missing}")

        # Fingerprint only the columns used (and the index)
        data_hash = self._hash_frame_cached(bars, spec.inputs)
        key = f"{symbol}__{spec.spec_hash(self.engine_version)}__{data_hash}"

        if self.enable_memory_cache and key in self._mem_cache:
            cached = self._mem_cache[key]
            meta = {"symbol": symbol, "spec": spec, "cache": "memory", "key": key}
            return cached, self._serialize_meta(meta)

        if self.enable_disk_cache and self.cache_dir:
            path = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    cached = pickle.load(f)
                if self.enable_memory_cache:
                    self._mem_cache[key] = cached
                meta = {"symbol": symbol, "spec": spec, "cache": "disk", "key": key}
                return cached, self._serialize_meta(meta)

        # Compute
        fn = _INDICATOR_REGISTRY[spec.indicator]
        raw = fn(bars, dict(spec.params))  # pass a copy

        df_feat = self._normalize_output(raw, bars.index, spec)
        df_feat = self._apply_warmup(df_feat, spec)

        # Save caches
        if self.enable_memory_cache:
            self._mem_cache[key] = df_feat
        if self.enable_disk_cache and self.cache_dir:
            path = os.path.join(self.cache_dir, f"{key}.pkl")
            with open(path, "wb") as f:
                pickle.dump(df_feat, f, protocol=pickle.HIGHEST_PROTOCOL)

        meta = {"symbol": symbol, "spec": spec, "cache": "miss", "key": key}
        return df_feat, self._serialize_meta(meta)

    def _hash_frame_cached(self, bars: pd.DataFrame, inputs: Tuple[str, ...]) -> str:
        """Memoize the expensive pandas hashing per (bars object, inputs tuple).

        During optimization you often compute dozens/hundreds of SMA windows. The raw
        price data (bars + inputs) is identical across specs, so hashing it once per
        inputs avoids O(n_specs) repeated hashing.
        """
        k = (id(bars), tuple(inputs))
        if k in self._framehash_cache:
            return self._framehash_cache[k]
        h = self._hash_frame(bars.loc[:, list(inputs)])
        self._framehash_cache[k] = h
        return h

    @staticmethod
    def _hash_frame(df: pd.DataFrame) -> str:
        # stable hash including index + values
        h_index = pd.util.hash_pandas_object(df.index.to_series(), index=False).values
        h_vals = pd.util.hash_pandas_object(df, index=False).values
        payload = np.concatenate([h_index, h_vals]).tobytes()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _normalize_output(
        raw: Union[pd.Series, pd.DataFrame],
        index: pd.DatetimeIndex,
        spec: FeatureSpec
    ) -> pd.DataFrame:
        base = spec.canonical_name()

        if isinstance(raw, pd.Series):
            s = raw.reindex(index)
            s.name = base
            return s.to_frame()

        if isinstance(raw, pd.DataFrame):
            df = raw.reindex(index)
            # If columns are unnamed or generic, prefix them with base
            if df.columns.nlevels == 1:
                df = df.copy()
                df.columns = [f"{base}__{c}" for c in df.columns]
            else:
                # avoid MultiIndex columns in features (keep them flat)
                df = df.copy()
                df.columns = [f"{base}__{'__'.join(map(str, tup))}" for tup in df.columns.to_list()]
            return df

        raise TypeError(f"Indicator '{spec.indicator}' returned unsupported type: {type(raw)}")

    @staticmethod
    def _apply_warmup(df_feat: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
        if spec.warmup is None:
            return df_feat
        df = df_feat.copy()
        df.iloc[: spec.warmup, :] = np.nan
        return df

    @staticmethod
    def _serialize_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
        spec: FeatureSpec = meta["spec"]
        return {
            "symbol": meta["symbol"],
            "indicator": spec.indicator,
            "params": dict(spec.params),
            "inputs": list(spec.inputs),
            "name": spec.name,
            "warmup": spec.warmup,
            "output_mode": spec.output_mode,
            "cache": meta["cache"],
            "key": meta["key"],
        }


# ---------- Built-in indicators (examples) ----------

@register_indicator("sma")
def sma(bars: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    window = int(params.get("window", 20))
    return bars["Close"].rolling(window=window, min_periods=window).mean()

@register_indicator("ema")
def ema(bars: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    span = int(params.get("span", 20))
    return bars["Close"].ewm(span=span, adjust=False, min_periods=span).mean()

@register_indicator("returns")
def returns(bars: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    method = params.get("method", "log")  # "log" or "simple"
    px = bars["Close"]
    if method == "simple":
        return px.pct_change()
    return np.log(px).diff()

@register_indicator("rsi")
def rsi(bars: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """
    Wilder RSI using exponential smoothing (alpha = 1/period).
    Output: RSI in [0, 100].
    """
    period = int(params.get("period", params.get("window", 14)))
    close = bars["Close"].astype(float)

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing is an EMA with alpha=1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


@register_indicator("macd")
def macd(bars: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    MACD line = EMA(close, fast) - EMA(close, slow)
    Signal    = EMA(MACD line, signal)
    Hist      = MACD line - Signal
    Output: DataFrame with columns: line, signal, hist
    """
    fast = int(params.get("fast", 12))
    slow = int(params.get("slow", 26))
    sig  = int(params.get("signal", 9))

    close = bars["Close"].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()

    line = ema_fast - ema_slow
    signal = line.ewm(span=sig, adjust=False, min_periods=sig).mean()
    hist = line - signal

    return pd.DataFrame({"line": line, "signal": signal, "hist": hist}, index=bars.index)

@register_indicator("rolling_std")
def rolling_std(df: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    w = int(params["window"])
    s = pd.to_numeric(df["Close"], errors="coerce")
    return s.rolling(w, min_periods=w).std().rename(f"std_{w}")
