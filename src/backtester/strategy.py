# strategy.py
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

# Import your FeatureSpec from the indicator layer.
# (Keep strategy -> indicators dependency; indicators should NOT depend on strategy.)
from indicators import FeatureSpec


# =============================================================================
# Protocols (duck-typing) to avoid tight coupling between layers
# =============================================================================

class MarketDataLike(Protocol):
    """
    Minimal interface expected from the data layer.
    """
    bars: Dict[str, pd.DataFrame]
    source: str
    timezone: str
    interval: str
    meta: Dict[str, Any]


class FeaturesDataLike(Protocol):
    """
    Minimal interface expected from the indicator layer.
    """
    features: Dict[str, pd.DataFrame]
    source: str
    timezone: str
    interval: str
    meta: Dict[str, Any]


# =============================================================================
# Strategy outputs (what Strategy layer returns downstream)
# =============================================================================

@dataclass
class SignalFrame:
    """
    Canonical output of a Strategy: "intent" at time t.

    signals:
        DataFrame indexed by timestamps, columns = symbols.
        Values represent desired exposure/intention at time t:
            - for long/flat strategies: {0, +1}
            - if allow_short: {-1, 0, +1}
        IMPORTANT: execution semantics (fill at t+1, costs, slippage) are NOT here.
        Those belong to portfolio/execution layers.

    validity:
        Optional DataFrame of same shape as signals, True where signal is valid
        (e.g., after warmup), False otherwise. Useful for debugging/reporting.

    meta:
        Strategy signature and any diagnostics.
    """
    signals: pd.DataFrame
    validity: Optional[pd.DataFrame] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def assert_well_formed(self, symbols: Sequence[str]) -> None:
        if not isinstance(self.signals.index, pd.DatetimeIndex):
            raise TypeError("SignalFrame.signals must be indexed by a DatetimeIndex")
        missing_cols = [s for s in symbols if s not in self.signals.columns]
        if missing_cols:
            raise ValueError(f"SignalFrame missing symbols: {missing_cols}")

        if self.validity is not None:
            if not self.validity.index.equals(self.signals.index):
                raise ValueError("SignalFrame.validity index must match signals index")
            if list(self.validity.columns) != list(self.signals.columns):
                raise ValueError("SignalFrame.validity columns must match signals columns")


# =============================================================================
# Strategy base class (contracts)
# =============================================================================

@dataclass(frozen=True)
class StrategySpec:
    """
    A stable, serializable description of a strategy configuration.
    Useful for logging, caching, and optimizer bookkeeping.
    """
    name: str
    params: Dict[str, Any]

    def signature(self) -> str:
        # Deterministic signature (stable key order)
        items = [(k, self.params[k]) for k in sorted(self.params)]
        return f"{self.name}|" + "|".join([f"{k}={v}" for k, v in items])


class BaseStrategy(ABC):
    """
    Strategy layer responsibility:
        - declare which features are required (FeatureSpec list)
        - generate "signals/intent" from MarketData + FeaturesData

    Strategy layer DOES NOT:
        - size positions in shares/weights
        - apply transaction costs/slippage
        - implement fill semantics (t vs t+1)
        - maintain trade ledger
    """

    @property
    @abstractmethod
    def spec(self) -> StrategySpec:
        """Stable description of the strategy (name + params)."""
        raise NotImplementedError

    @abstractmethod
    def required_features(self) -> List[FeatureSpec]:
        """
        Return the list of FeatureSpec needed by this strategy.

        This allows the engine (or research runner) to compute indicators
        upstream in a consistent way.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_signals(
        self,
        market_data: MarketDataLike,
        features_data: FeaturesDataLike,
        symbols: Optional[Sequence[str]] = None,
    ) -> SignalFrame:
        """
        Vectorized/batch implementation (research-first).
        Returns desired exposures at time t.
        """
        raise NotImplementedError

    # Optional extension point for future event-driven parity
    def on_bar(
        self,
        t: pd.Timestamp,
        symbol: str,
        bar_t: pd.Series,
        features_t: pd.Series,
        state: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Optional incremental interface (not required for your current research-first engine).
        Returns desired exposure at time t for one symbol.
        """
        raise NotImplementedError("on_bar not implemented for this strategy.")

# --- Buy & Hold -------------------------------------------------

@dataclass(frozen=True)
class BuyHoldParams:
    buy_pct_cash: float = 1.0
    nan_policy: str = "flat"  # "flat" or "nan"


class BuyHoldStrategy(BaseStrategy):
    def __init__(self, params: BuyHoldParams) -> None:
        self.params = params

    @property
    def spec(self) -> StrategySpec:
        return StrategySpec(name="BuyHoldStrategy", params=asdict(self.params))

    def required_features(self) -> List[FeatureSpec]:
        # no indicators needed
        return []

    def generate_signals(
        self,
        market_data: MarketDataLike,
        features_data: FeaturesDataLike,
        symbols: Optional[Sequence[str]] = None,
    ) -> SignalFrame:

        symbols = list(symbols) if symbols is not None else list(market_data.bars.keys())

        # build a common index (union) consistent with your other strategies
        all_indexes = [market_data.bars[s].index for s in symbols]
        common_index = all_indexes[0]
        for idx in all_indexes[1:]:
            common_index = common_index.union(idx)
        common_index = common_index.sort_values()

        sig_df = pd.DataFrame(index=common_index, columns=symbols, dtype="float64")
        valid_df = pd.DataFrame(index=common_index, columns=symbols, dtype="bool")

        buy_pct = float(self.params.buy_pct_cash)

        for s in symbols:
            bars = market_data.bars[s].reindex(common_index)
            close = pd.to_numeric(bars["Close"], errors="coerce")

            valid = close.notna()
            out = pd.Series(0.0, index=common_index, dtype=float)

            # buy once on first valid bar and hold
            if valid.any():
                first_idx = valid.idxmax()  # first True
                out.loc[first_idx:] = 1.0 * buy_pct

            if self.params.nan_policy == "flat":
                out = out.where(valid, 0.0)
            else:
                out = out.where(valid, np.nan)

            sig_df[s] = out
            valid_df[s] = valid

        meta = {
            "strategy_signature": self.spec.signature(),
            "strategy_name": self.spec.name,
            "strategy_params": self.spec.params,
            "required_features": [],
        }

        sf = SignalFrame(signals=sig_df, validity=valid_df, meta=meta)
        sf.assert_well_formed(symbols)
        return sf

# =============================================================================
# Concrete strategy: Moving Average Crossover (MA crossover)
# =============================================================================

@dataclass(frozen=True)
class MovingAverageCrossParams:
    """
    Parameters for MA crossover.

    fast_window, slow_window:
        SMA windows in bars.

    allow_short:
        If True, emit -1 when fast < slow.
        If False, emit 0 when fast < slow (long/flat).

    nan_policy:
        How to treat periods where the MAs are undefined (warmup).
        - "flat": output 0
        - "nan": output NaN (downstream must decide)
    """
    fast_window: int = 15
    slow_window: int = 50
    allow_short: bool = False
    nan_policy: str = "flat"  # "flat" or "nan"

    def __post_init__(self) -> None:
        if self.fast_window <= 0 or self.slow_window <= 0:
            raise ValueError("fast_window and slow_window must be positive")
        if self.fast_window >= self.slow_window:
            raise ValueError("Require fast_window < slow_window for crossover")
        if self.nan_policy not in ("flat", "nan"):
            raise ValueError("nan_policy must be 'flat' or 'nan'")


class MovingAverageCrossStrategy(BaseStrategy):
    """
    MA crossover strategy producing desired exposure at time t.

    Signal rule:
        long if SMA_fast[t] > SMA_slow[t]
        else flat (0) or short (-1) depending on allow_short

    Note: this strategy only generates intent; execution semantics are downstream.
    """

    def __init__(self, params: MovingAverageCrossParams) -> None:
        self.params = params

        # We choose deterministic feature names so the strategy can locate columns reliably.
        self._fast_name = f"sma_{params.fast_window}"
        self._slow_name = f"sma_{params.slow_window}"

    @property
    def spec(self) -> StrategySpec:
        return StrategySpec(
            name="MovingAverageCrossStrategy",
            params=asdict(self.params),
        )

    def required_features(self) -> List[FeatureSpec]:
        # Warmup is set to the window length; conservative and explicit.
        # Strategy will treat pre-warmup according to nan_policy.
        return [
            FeatureSpec(
                indicator="sma",
                params={"window": self.params.fast_window},
                inputs=("Close",),
                warmup=self.params.fast_window,
                
            ),
            FeatureSpec(
                indicator="sma",
                params={"window": self.params.slow_window},
                inputs=("Close",),
                warmup=self.params.slow_window,
                
            ),
        ]

    def generate_signals(
        self,
        market_data: MarketDataLike,
        features_data: FeaturesDataLike,
        symbols: Optional[Sequence[str]] = None,
    ) -> SignalFrame:

        symbols = list(symbols) if symbols is not None else list(market_data.bars.keys())

        # Validate presence of features for requested symbols
        for s in symbols:
            if s not in features_data.features:
                raise KeyError(f"FeaturesData missing symbol '{s}'. Computed: {list(features_data.features.keys())}")
            if s not in market_data.bars:
                raise KeyError(f"MarketData missing symbol '{s}'. Available: {list(market_data.bars.keys())}")

        # Build a common index per symbol (we keep per-symbol for now; later you can align multi-asset)
        # Output as a single DataFrame: index is the union of indexes, reindexed per symbol.
        # For single-asset this is trivial.
        all_indexes = [market_data.bars[s].index for s in symbols]
        common_index = all_indexes[0]
        for idx in all_indexes[1:]:
            common_index = common_index.union(idx)
        common_index = common_index.sort_values()

        sig_df = pd.DataFrame(index=common_index, columns=symbols, dtype="float64")
        valid_df = pd.DataFrame(index=common_index, columns=symbols, dtype="bool")

        for s in symbols:
            bars = market_data.bars[s]
            feats = features_data.features[s]

            # Reindex features to the common index
            fast = feats[self._fast_name].reindex(common_index)
            slow = feats[self._slow_name].reindex(common_index)

            # Valid when both are not NaN
            valid = fast.notna() & slow.notna()

            # Core crossover logic
            long_mask = fast > slow
            short_mask = fast < slow

            signal = pd.Series(0.0, index=common_index)
            signal[long_mask] = 1.0
            signal[short_mask] = -1.0


            # Apply NaN/warmup policy
            if self.params.nan_policy == "flat":
                # Before validity, stay flat (0) to avoid accidental early exposure
                signal = signal.where(valid, 0.0)
            else:  # "nan"
                signal = signal.where(valid, np.nan)

            sig_df[s] = signal
            valid_df[s] = valid

        meta = {
            "strategy_signature": self.spec.signature(),
            "strategy_name": self.spec.name,
            "strategy_params": self.spec.params,
            "required_features": [fs.canonical_name() for fs in self.required_features()],
            "note": "signals are intent at time t; fills/costs handled downstream",
        }

        sf = SignalFrame(signals=sig_df, validity=valid_df, meta=meta)
        sf.assert_well_formed(symbols)
        return sf


# =============================================================================
# Concrete strategy: Price vs SMA (single moving average filter)
# =============================================================================

@dataclass(frozen=True)
class PriceAboveSMAParams:
    window: int = 50
    allow_short: bool = False
    nan_policy: str = "flat"  # "flat" or "nan"

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.nan_policy not in ("flat", "nan"):
            raise ValueError("nan_policy must be 'flat' or 'nan'")


class PriceAboveSMAStrategy(BaseStrategy):
    """
    Signal rule at time t:
        if Close[t] > SMA_window[t] -> +1
        elif allow_short and Close[t] < SMA_window[t] -> -1
        else -> 0
    """

    def __init__(self, params: PriceAboveSMAParams) -> None:
        self.params = params
        self._sma_name = f"sma_{params.window}"

    @property
    def spec(self) -> StrategySpec:
        return StrategySpec(
            name="PriceAboveSMAStrategy",
            params=asdict(self.params),
        )

    def required_features(self) -> List[FeatureSpec]:
        return [
            FeatureSpec(
                indicator="sma",
                params={"window": self.params.window},
                inputs=("Close",),
                warmup=self.params.window,
            )
        ]

    def generate_signals(
        self,
        market_data: MarketDataLike,
        features_data: FeaturesDataLike,
        symbols: Optional[Sequence[str]] = None,
    ) -> SignalFrame:
        symbols = list(symbols) if symbols is not None else list(market_data.bars.keys())

        # common index
        all_indexes = [market_data.bars[s].index for s in symbols]
        common_index = all_indexes[0]
        for idx in all_indexes[1:]:
            common_index = common_index.union(idx)
        common_index = common_index.sort_values()

        sig_df = pd.DataFrame(index=common_index, columns=symbols, dtype="float64")
        valid_df = pd.DataFrame(index=common_index, columns=symbols, dtype="bool")

        for s in symbols:
            bars = market_data.bars[s]
            feats = features_data.features[s]

            close = bars["Close"].reindex(common_index).astype(float)
            sma = feats[self._sma_name].reindex(common_index).astype(float)

            valid = close.notna() & sma.notna()
            long_mask = close > sma
            short_mask = close < sma  # treat as EXIT when long-only

            signal = pd.Series(0.0, index=common_index)
            signal[long_mask] = 1.0
            signal[short_mask] = -1.0  # <-- emit exit signal even if allow_short=False

            # If allow_short True, -1 can mean actual short exposure downstream,
            # but in long-only target_weight mode it's clamped to 0 anyway.
            if self.params.nan_policy == "flat":
                signal = signal.where(valid, 0.0)
            else:
                signal = signal.where(valid, np.nan)


            if self.params.nan_policy == "flat":
                signal = signal.where(valid, 0.0)
            else:
                signal = signal.where(valid, np.nan)

            sig_df[s] = signal
            valid_df[s] = valid

        meta = {
            "strategy_signature": self.spec.signature(),
            "strategy_name": self.spec.name,
            "strategy_params": self.spec.params,
            "required_features": [fs.canonical_name() for fs in self.required_features()],
            "note": "signals are intent at time t; fills/costs handled downstream",
        }

        sf = SignalFrame(signals=sig_df, validity=valid_df, meta=meta)
        sf.assert_well_formed(symbols)
        return sf

@dataclass(frozen=True)
class RSIParams:
    period: int = 12
    low: float = 30.0
    high: float = 70.0
    mode: str = "reversal"   # "reversal" or "momentum"
    allow_short: bool = False
    nan_policy: str = "flat"

    def __post_init__(self) -> None:
        if self.period <= 0:
            raise ValueError("period must be positive")
        if not (0.0 <= self.low < self.high <= 100.0):
            raise ValueError("Require 0 <= low < high <= 100")
        if self.mode not in ("reversal", "momentum"):
            raise ValueError("mode must be 'reversal' or 'momentum'")
        if self.nan_policy not in ("flat", "nan"):
            raise ValueError("nan_policy must be 'flat' or 'nan'")


class RSIStrategy(BaseStrategy):
    def __init__(self, params: RSIParams) -> None:
        self.params = params
        self._rsi_name = f"rsi_{params.period}"

    @property
    def spec(self) -> StrategySpec:
        return StrategySpec(name="RSIStrategy", params=asdict(self.params))

    def required_features(self) -> List[FeatureSpec]:
        return [
            FeatureSpec(
                indicator="rsi",
                params={"period": self.params.period},
                inputs=("Close",),
                warmup=self.params.period,
            )
        ]

    def generate_signals(
        self,
        market_data: MarketDataLike,
        features_data: FeaturesDataLike,
        symbols: Optional[Sequence[str]] = None,
    ) -> SignalFrame:
        symbols = list(symbols) if symbols is not None else list(market_data.bars.keys())

        all_indexes = [market_data.bars[s].index for s in symbols]
        common_index = all_indexes[0]
        for idx in all_indexes[1:]:
            common_index = common_index.union(idx)
        common_index = common_index.sort_values()

        sig_df = pd.DataFrame(index=common_index, columns=symbols, dtype="float64")
        valid_df = pd.DataFrame(index=common_index, columns=symbols, dtype="bool")

        for s in symbols:
            r = features_data.features[s][self._rsi_name].reindex(common_index).astype(float)
            valid = r.notna()

            signal = pd.Series(0.0, index=common_index, dtype=float)

            if self.params.mode == "reversal":
                # long when oversold, exit when overbought
                signal[r < self.params.low] = 1.0
                signal[r > self.params.high] = -1.0
            else:  # momentum
                # long when strong, exit when weak
                signal[r > self.params.high] = 1.0
                signal[r < self.params.low] = -1.0

            if self.params.nan_policy == "flat":
                signal = signal.where(valid, 0.0)
            else:
                signal = signal.where(valid, np.nan)

            sig_df[s] = signal
            valid_df[s] = valid

        meta = {
            "strategy_signature": self.spec.signature(),
            "strategy_name": self.spec.name,
            "strategy_params": self.spec.params,
            "required_features": [fs.canonical_name() for fs in self.required_features()],
        }

        sf = SignalFrame(signals=sig_df, validity=valid_df, meta=meta)
        sf.assert_well_formed(symbols)
        return sf

@dataclass(frozen=True)
class MACDParams:
    fast: int = 12
    slow: int = 26
    signal: int = 9
    trigger: str = "zero"   # "cross" or "zero"
    allow_short: bool = False
    nan_policy: str = "flat"

    def __post_init__(self) -> None:
        if self.fast <= 0 or self.slow <= 0 or self.signal <= 0:
            raise ValueError("fast/slow/signal must be positive")
        if self.fast >= self.slow:
            raise ValueError("Require fast < slow")
        if self.trigger not in ("cross", "zero"):
            raise ValueError("trigger must be 'cross' or 'zero'")
        if self.nan_policy not in ("flat", "nan"):
            raise ValueError("nan_policy must be 'flat' or 'nan'")


class MACDStrategy(BaseStrategy):
    def __init__(self, params: MACDParams) -> None:
        self.params = params
        self._base = f"macd_{params.fast}_{params.slow}_{params.signal}"
        self._line_col = f"{self._base}__line"
        self._sig_col = f"{self._base}__signal"

    @property
    def spec(self) -> StrategySpec:
        return StrategySpec(name="MACDStrategy", params=asdict(self.params))

    def required_features(self) -> List[FeatureSpec]:
        warmup = int(self.params.slow + self.params.signal)  # conservative
        return [
            FeatureSpec(
                indicator="macd",
                params={"fast": self.params.fast, "slow": self.params.slow, "signal": self.params.signal},
                inputs=("Close",),
                warmup=warmup,
            )
        ]

    def generate_signals(
        self,
        market_data: MarketDataLike,
        features_data: FeaturesDataLike,
        symbols: Optional[Sequence[str]] = None,
    ) -> SignalFrame:
        symbols = list(symbols) if symbols is not None else list(market_data.bars.keys())

        all_indexes = [market_data.bars[s].index for s in symbols]
        common_index = all_indexes[0]
        for idx in all_indexes[1:]:
            common_index = common_index.union(idx)
        common_index = common_index.sort_values()

        sig_df = pd.DataFrame(index=common_index, columns=symbols, dtype="float64")
        valid_df = pd.DataFrame(index=common_index, columns=symbols, dtype="bool")

        for s in symbols:
            feats = features_data.features[s]
            line = feats[self._line_col].reindex(common_index).astype(float)
            sigl = feats[self._sig_col].reindex(common_index).astype(float)

            valid = line.notna() & sigl.notna()
            out = pd.Series(0.0, index=common_index, dtype=float)

            if self.params.trigger == "cross":
                prev_line = line.shift(1)
                prev_sigl = sigl.shift(1)

                buy = (line > sigl) & (prev_line <= prev_sigl)   # cross up
                sell = (line < sigl) & (prev_line >= prev_sigl)  # cross down

                out[:] = 0.0
                out[buy] = 1.0
                out[sell] = -1.0

            else:  # "zero"
                out[line > 0.0] = 1.0
                out[line < 0.0] = -1.0
                prev_line = line.shift(1)
                prev_sigl = sigl.shift(1)

                buy = (line > 0.0) & (prev_line <= 0)   # cross up
                sell = (line < 0.0) & (prev_line >= 0)  # cross down

                out[:] = 0.0
                out[buy] = 1.0
                out[sell] = -1.0

            if self.params.nan_policy == "flat":
                out = out.where(valid, 0.0)
            else:
                out = out.where(valid, np.nan)

            sig_df[s] = out
            valid_df[s] = valid

        meta = {
            "strategy_signature": self.spec.signature(),
            "strategy_name": self.spec.name,
            "strategy_params": self.spec.params,
            "required_features": [fs.canonical_name() for fs in self.required_features()],
        }

        sf = SignalFrame(signals=sig_df, validity=valid_df, meta=meta)
        sf.assert_well_formed(symbols)
        return sf

# =============================================================================
# Concrete strategy: Bollinger Bands (mean reversion)
# =============================================================================

@dataclass(frozen=True)
class BollingerParams:
    bb_window: int = 20
    bb_k: float = 2.0
    allow_short: bool = False
    nan_policy: str = "flat"

    def __post_init__(self) -> None:
        if self.bb_window <= 0:
            raise ValueError("bb_window must be positive")
        if self.bb_k <= 0:
            raise ValueError("bb_k must be positive")
        if self.nan_policy not in ("flat", "nan"):
            raise ValueError("nan_policy must be 'flat' or 'nan'")



class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, params: BollingerParams) -> None:
        self.params = params
        w = int(params.bb_window)
        self._mid = f"sma_{w}"
        self._std = f"std_{w}"

    @property
    def spec(self) -> StrategySpec:
        return StrategySpec(name="BollingerBandsStrategy", params=asdict(self.params))

    def required_features(self) -> List[FeatureSpec]:
        w = int(self.params.bb_window)
        return [
            FeatureSpec(
                indicator="sma",
                params={"window": w},
                inputs=("Close",),
                name=f"sma_{w}",     # explicit column name
                warmup=w,
                output_mode="series",
            ),
            FeatureSpec(
                indicator="rolling_std",          # ✅ MUST match your registry key
                params={"window": w},
                inputs=("Close",),
                name=f"std_{w}",                  # explicit column name
                warmup=w,
                output_mode="series",
            ),
        ]

    def generate_signals(
        self,
        market_data: MarketDataLike,
        features_data: FeaturesDataLike,
        symbols: Optional[Sequence[str]] = None,
    ) -> SignalFrame:

        symbols = list(symbols) if symbols is not None else list(market_data.bars.keys())

        # common index
        all_indexes = [market_data.bars[s].index for s in symbols]
        common_index = all_indexes[0]
        for idx in all_indexes[1:]:
            common_index = common_index.union(idx)
        common_index = common_index.sort_values()

        sig_df = pd.DataFrame(index=common_index, columns=symbols, dtype="float64")
        valid_df = pd.DataFrame(index=common_index, columns=symbols, dtype="bool")

        w = int(self.params.bb_window)
        k = float(self.params.bb_k)

        for s in symbols:
            bars = market_data.bars[s]
            feats = features_data.features[s].reindex(common_index)

            close = pd.to_numeric(bars["Close"].reindex(common_index), errors="coerce")
            mid = pd.to_numeric(feats[self._mid], errors="coerce")
            std = pd.to_numeric(feats[self._std], errors="coerce")

            upper = mid + k * std
            lower = mid - k * std

            valid = close.notna() & mid.notna() & std.notna()

            out = pd.Series(0.0, index=common_index, dtype=float)
            out[close < lower] = 1.0
            out[close > upper] = -1.0

            if not self.params.allow_short:
                # In long/flat mode, treat -1 as exit/flat intent
                # (portfolio layer may clamp anyway, but keep intent clean)
                out[out < 0] = -1.0  # keep as exit signal if you want; or set to 0.0
                # If you prefer strict flat instead of "exit" semantics, do:
                # out[out < 0] = 0.0

            if self.params.nan_policy == "flat":
                out = out.where(valid, 0.0)
            else:
                out = out.where(valid, np.nan)

            sig_df[s] = out
            valid_df[s] = valid

        meta = {
            "strategy_signature": self.spec.signature(),
            "strategy_name": self.spec.name,
            "strategy_params": self.spec.params,
            "required_features": [fs.canonical_name() for fs in self.required_features()],
        }

        sf = SignalFrame(signals=sig_df, validity=valid_df, meta=meta)
        sf.assert_well_formed(symbols)
        return sf


# =============================================================================
# Helpers for engine/UI: plot overlay indicators per strategy
# =============================================================================

from typing import Any, Dict, List

def default_plot_indicators(cfg_kind: str, cfg_params: Dict[str, Any]) -> List[str]:
    k = (cfg_kind or "").lower()
    p = cfg_params or {}

    # -------------------------
    # Existing strategies
    # -------------------------
    if k == "ma_cross":
        fast = int(p.get("sma_fast_window", p.get("fast_window", 20)))
        slow = int(p.get("sma_slow_window", p.get("slow_window", 50)))
        return [f"sma_{fast}", f"sma_{slow}"]

    if k in ("sma_price", "price_sma", "price_above_sma"):
        w = int(p.get("sma_window", p.get("window", 50)))
        return [f"sma_{w}"]

    if k == "rsi":
        n = int(p.get("rsi_window", p.get("period", 14)))
        return [f"rsi_{n}"]

    if k == "macd":
        fast = int(p.get("macd_fast_window", p.get("fast", 12)))
        slow = int(p.get("macd_slow_window", p.get("slow", 26)))
        sig  = int(p.get("macd_signal_window", p.get("signal", 9)))
        base = f"macd_{fast}_{slow}_{sig}"
        # include hist too if you compute it
        return [f"{base}__line", f"{base}__signal", f"{base}__hist"]

    if k == "bollinger":
        w = int(p.get("bb_window", 20))
        return [f"sma_{w}", f"std_{w}"]

    # -------------------------
    # NEW strategies
    # -------------------------

    # === OBV ===
    # Indicators expected by your plot:
    #   "obv", "obv_ema_{span}"
    if k == "obv":
        span = int(p.get("obv_span", p.get("span", 20)))
        return ["obv", f"obv_ema_{span}"]

    # === Stoch + VWAP ===
    # Indicators expected by your plot:
    #   stoch_{k}_{d}_{smooth}__k
    #   stoch_{k}_{d}_{smooth}__d
    # and optionally vwap_{window} on price row
    if k in ("stoch_vwap", "stoch", "stochastic"):
        k_w = int(p.get("k_window", 14))
        d_w = int(p.get("d_window", 3))
        s_k = int(p.get("smooth_k", 1))
        v_w = int(p.get("vwap_window", 20))

        base = f"stoch_{k_w}_{d_w}_{s_k}"
        return [f"{base}__k", f"{base}__d", f"vwap_{v_w}"]

    # === Ichimoku ===
    # Indicators expected by your plot:
    #   ichimoku_{ten}_{kij}_{sb}_{shift}__tenkan/kijun/span_a/span_b
    if k == "ichimoku":
        ten = int(p.get("tenkan", 9))
        kij = int(p.get("kijun", 26))
        sb  = int(p.get("senkou_b", 52))
        sh  = int(p.get("shift", 26))

        base = f"ichimoku_{ten}_{kij}_{sb}_{sh}"
        return [
            f"{base}__tenkan",
            f"{base}__kijun",
            f"{base}__span_a",
            f"{base}__span_b",
        ]

    return []

@dataclass(frozen=True)
class OBVParams:
    obv_span: int = 20
    allow_short: bool = False
    nan_policy: str = "flat"  # "flat" or "nan"

class OBVStrategy(BaseStrategy):
    def __init__(self, params: OBVParams) -> None:
        self.params = params
        self._obv_col = "obv"
        # EMA is computed on Close in your current registry,
        # so we will compute EMA(OBV) ourselves in strategy (institutionally: keep indicator pure).
        # To keep everything in indicator engine, we instead compute a dedicated EMA on OBV by reusing pandas here.
        self._ema_span = int(params.obv_span)

    @property
    def spec(self) -> StrategySpec:
        return StrategySpec(name="OBVStrategy", params=asdict(self.params))

    def required_features(self) -> List[FeatureSpec]:
        return [
            FeatureSpec(indicator="obv", params={}, inputs=("Close", "Volume"), warmup=2),
        ]

    def generate_signals(self, market_data: MarketDataLike, features_data: FeaturesDataLike,
                         symbols: Optional[Sequence[str]] = None) -> SignalFrame:
        symbols = list(symbols) if symbols is not None else list(market_data.bars.keys())

        # union index (consistent with your other strategies)
        common_index = market_data.bars[symbols[0]].index
        for s in symbols[1:]:
            common_index = common_index.union(market_data.bars[s].index)
        common_index = common_index.sort_values()

        sig_df = pd.DataFrame(index=common_index, columns=symbols, dtype="float64")
        valid_df = pd.DataFrame(index=common_index, columns=symbols, dtype="bool")

        for s in symbols:
            obv = features_data.features[s][self._obv_col].reindex(common_index).astype(float)
            obv_ema = obv.ewm(span=self._ema_span, adjust=False, min_periods=self._ema_span).mean()

            valid = obv.notna() & obv_ema.notna()
            signal = pd.Series(0.0, index=common_index, dtype=float)

            signal[obv > obv_ema] = 1.0
            if self.params.allow_short:
                signal[obv < obv_ema] = -1.0
            else:
                # long/flat: exit when condition fails
                signal[obv < obv_ema] = -1.0  # “sell/exit intent”, PortfolioEngine will reduce exposure

            if self.params.nan_policy == "flat":
                signal = signal.where(valid, 0.0)
            else:
                signal = signal.where(valid, np.nan)

            sig_df[s] = signal
            valid_df[s] = valid

        return SignalFrame(
            signals=sig_df,
            validity=valid_df,
            meta={"strategy_signature": self.spec.signature(), "note": "OBV vs EMA(OBV)"}
        )

@dataclass(frozen=True)
class StochVWAPParams:
    k_window: int = 14
    d_window: int = 3
    smooth_k: int = 1
    vwap_window: int = 20
    allow_short: bool = False
    nan_policy: str = "flat"  # "flat" or "nan"
class StochVWAPStrategy(BaseStrategy):
    def __init__(self, params: StochVWAPParams) -> None:
        self.params = params
        self._stoch_base = f"stoch_{params.k_window}_{params.d_window}_{params.smooth_k}"
        self._k_col = f"{self._stoch_base}__k"
        self._d_col = f"{self._stoch_base}__d"
        self._vwap_col = f"vwap_{params.vwap_window}"

    @property
    def spec(self) -> StrategySpec:
        return StrategySpec(name="StochVWAPStrategy", params=asdict(self.params))

    def required_features(self) -> List[FeatureSpec]:
        warm = int(max(self.params.k_window, self.params.d_window, self.params.smooth_k, self.params.vwap_window) + 1)
        return [
            FeatureSpec(
                indicator="stoch",
                params={"k_window": self.params.k_window, "d_window": self.params.d_window, "smooth_k": self.params.smooth_k},
                inputs=("High", "Low", "Close"),
                warmup=warm,
            ),
            FeatureSpec(
                indicator="vwap",
                params={"window": self.params.vwap_window},
                inputs=("High", "Low", "Close", "Volume"),
                warmup=self.params.vwap_window,
            ),
        ]

    @staticmethod
    def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a.shift(1) <= b.shift(1)) & (a > b)

    @staticmethod
    def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a.shift(1) >= b.shift(1)) & (a < b)

    def generate_signals(self, market_data: MarketDataLike, features_data: FeaturesDataLike,
                         symbols: Optional[Sequence[str]] = None) -> SignalFrame:
        symbols = list(symbols) if symbols is not None else list(market_data.bars.keys())

        common_index = market_data.bars[symbols[0]].index
        for s in symbols[1:]:
            common_index = common_index.union(market_data.bars[s].index)
        common_index = common_index.sort_values()

        sig_df = pd.DataFrame(index=common_index, columns=symbols, dtype="float64")
        valid_df = pd.DataFrame(index=common_index, columns=symbols, dtype="bool")

        for s in symbols:
            bars = market_data.bars[s].reindex(common_index)
            close = bars["Close"].astype(float)

            feats = features_data.features[s].reindex(common_index)
            k = feats[self._k_col].astype(float)
            d = feats[self._d_col].astype(float)
            vwap = feats[self._vwap_col].astype(float)

            valid = k.notna() & d.notna() & vwap.notna() & close.notna()

            kx_up = self._cross_up(k, d)
            kx_dn = self._cross_down(k, d)
            cx_up = self._cross_up(close, vwap)
            cx_dn = self._cross_down(close, vwap)

            # BUY conditions
            buy1 = kx_up & (k < 20) & (d < 20) & (close > vwap)
            buy2 = (k > 50) & (k < 80) & (d > 50) & (d < 80) & cx_up
            buy3 = kx_up & (k < 80) & (d < 80) & (close > vwap)
            buy = buy1 | buy2 | buy3

            # SELL conditions
            sell1 = kx_dn & (k > 80) & (d > 80) & (close < vwap)
            sell2 = (k > 20) & (k < 50) & (d > 20) & (d < 50) & cx_dn
            sell3 = kx_dn & (k > 20) & (d > 20) & (close < vwap)
            sell = sell1 | sell2 | sell3

            # Precedence: SELL > BUY
            signal = pd.Series(0.0, index=common_index, dtype=float)
            signal[buy] = 1.0
            if self.params.allow_short:
                signal[sell] = -1.0
            else:
                # still emit -1 as “exit intent”
                signal[sell] = -1.0

            if self.params.nan_policy == "flat":
                signal = signal.where(valid, 0.0)
            else:
                signal = signal.where(valid, np.nan)

            sig_df[s] = signal
            valid_df[s] = valid

        return SignalFrame(
            signals=sig_df,
            validity=valid_df,
            meta={"strategy_signature": self.spec.signature(), "note": "Stoch+VWAP 3-condition logic (sell precedence)"}
        )


@dataclass(frozen=True)
class IchimokuParams:
    tenkan: int = 9
    kijun: int = 26
    senkou_b: int = 52
    shift: int = 26
    allow_short: bool = False
    nan_policy: str = "flat"  # "flat" or "nan"
class IchimokuStrategy(BaseStrategy):
    def __init__(self, params: IchimokuParams) -> None:
        self.params = params
        self._base = f"ichimoku_{params.tenkan}_{params.kijun}_{params.senkou_b}_{params.shift}"
        self._tenkan = f"{self._base}__tenkan"
        self._kijun = f"{self._base}__kijun"
        self._span_a = f"{self._base}__span_a"
        self._span_b = f"{self._base}__span_b"

    @property
    def spec(self) -> StrategySpec:
        return StrategySpec(name="IchimokuStrategy", params=asdict(self.params))

    def required_features(self) -> List[FeatureSpec]:
        warm = int(self.params.senkou_b + self.params.shift)  # conservative
        return [
            FeatureSpec(
                indicator="ichimoku",
                params={
                    "tenkan": self.params.tenkan,
                    "kijun": self.params.kijun,
                    "senkou_b": self.params.senkou_b,
                    "shift": self.params.shift,
                },
                inputs=("High", "Low", "Close"),
                warmup=warm,
            )
        ]

    def generate_signals(self, market_data: MarketDataLike, features_data: FeaturesDataLike,
                         symbols: Optional[Sequence[str]] = None) -> SignalFrame:
        symbols = list(symbols) if symbols is not None else list(market_data.bars.keys())

        common_index = market_data.bars[symbols[0]].index
        for s in symbols[1:]:
            common_index = common_index.union(market_data.bars[s].index)
        common_index = common_index.sort_values()

        sig_df = pd.DataFrame(index=common_index, columns=symbols, dtype="float64")
        valid_df = pd.DataFrame(index=common_index, columns=symbols, dtype="bool")

        for s in symbols:
            bars = market_data.bars[s].reindex(common_index)
            close = bars["Close"].astype(float)

            feats = features_data.features[s].reindex(common_index)
            tenkan = feats[self._tenkan].astype(float)
            kijun = feats[self._kijun].astype(float)
            span_a = feats[self._span_a].astype(float)
            span_b = feats[self._span_b].astype(float)

            cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
            cloud_bot = pd.concat([span_a, span_b], axis=1).min(axis=1)

            valid = close.notna() & tenkan.notna() & kijun.notna() & cloud_top.notna() & cloud_bot.notna()

            buy = (close > cloud_top) & (tenkan > kijun)
            sell = (close < cloud_bot) & (tenkan < kijun)

            signal = pd.Series(0.0, index=common_index, dtype=float)
            signal[buy] = 1.0
            if self.params.allow_short:
                signal[sell] = -1.0
            else:
                signal[sell] = -1.0  # exit intent

            if self.params.nan_policy == "flat":
                signal = signal.where(valid, 0.0)
            else:
                signal = signal.where(valid, np.nan)

            sig_df[s] = signal
            valid_df[s] = valid

        return SignalFrame(
            signals=sig_df,
            validity=valid_df,
            meta={"strategy_signature": self.spec.signature(), "note": "Kumo breakout + Tenkan/Kijun confirmation"}
        )
