from __future__ import annotations

from data import YahooFinanceDataSource   # your class
from data import MarketData
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Callable, Literal, Union
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import replace
import pandas as pd
import copy
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
# ---- Project modules (adapt import paths if you have a package folder) ----
from data import MarketData
from data import BMCEDataSource
from indicators import IndicatorEngine, FeatureSpec, FeaturesData
from strategy import (
    BaseStrategy,
    MovingAverageCrossStrategy,
    MovingAverageCrossParams,
    SignalFrame,
    PriceAboveSMAStrategy,
    PriceAboveSMAParams,
    RSIStrategy,
    RSIParams,
    MACDStrategy,
    MACDParams,
    BollingerBandsStrategy,
    BollingerParams,
    BuyHoldStrategy,
    BuyHoldParams,
    OBVStrategy, OBVParams,
    StochVWAPStrategy, StochVWAPParams,
    IchimokuStrategy, IchimokuParams,
)
from portfolio import PortfolioEngine, PortfolioConfig, PortfolioResult
from results import ResultsAnalyzer, BacktestReport
from strategy import default_plot_indicators  # add import


DataSourceKind = Literal["bmce", "yfinance"]
StrategyKind = Literal["ma_cross","sma_price", "rsi","macd","bollinger", "buy_hold","obv","stoch_vwap","ichimoku"]  # add more as you implement them'"]
def slice_df_by_start_end(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.sort_index()
    if start:
        out = out.loc[pd.to_datetime(start):]
    if end:
        out = out.loc[:pd.to_datetime(end)]
    return out


def slice_marketdata(md: MarketData, start: Optional[str], end: Optional[str]) -> MarketData:
    if md is None:
        return md
    new_bars = {sym: slice_df_by_start_end(df, start, end) for sym, df in md.bars.items()}
    return MarketData(bars=new_bars, source=md.source, timezone=md.timezone, interval=md.interval)


def slice_features(feats: FeaturesData, md: MarketData) -> FeaturesData:
    # align feats strictly to md index per symbol
    new = {}
    for sym, bars in md.bars.items():
        f = feats.features.get(sym, pd.DataFrame(index=bars.index))
        new[sym] = f.reindex(bars.index)
    return FeaturesData(features=new, source=feats.source, timezone=feats.timezone, interval=feats.interval, meta=feats.meta)

# -----------------------------
# Engine Spec Objects
# -----------------------------
@dataclass(frozen=True)
class DataConfig:
    source: DataSourceKind
    symbols: List[str]

    timezone: str = "GMT"
    interval: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
        # Period filters (multi-window)
    include_windows: Optional[List[tuple[str, str]]] = None
    exclude_windows: Optional[List[tuple[str, str]]] = None


    # BMCE inputs:
    # - for single symbol: str/Path
    # - for multi symbols: dict {symbol: str/Path}
    bmce_paths: Optional[Union[str, Path, Dict[str, Union[str, Path]]]] = None

    # yfinance inputs
    yf_period: str = "max"
    yf_interval: str = "1d"
    yf_auto_adjust: bool = False


@dataclass(frozen=True)
class IndicatorsConfig:
    """
    You can supply:
      - specs directly, OR
      - a builder that returns specs, OR
      - leave empty and let the engine infer specs from strategy kind (only for known strategies)
    """
    specs: Optional[List[FeatureSpec]] = None
    spec_builder: Optional[Callable[[], List[FeatureSpec]]] = None
    
    cache_dir: Optional[str] = ".cache/features"
    enable_disk_cache: bool = True
    enable_memory_cache: bool = True
    engine_version: str = "v1"


@dataclass(frozen=True)
class StrategyConfig:
    kind: StrategyKind
    params: Dict[str, Any] = field(default_factory=dict)

BenchmarkSource = Literal["yahoo"]

@dataclass(frozen=True)
class BenchmarkConfig:
    enabled: bool = False
    symbol: str = "SPY"
    source: BenchmarkSource = "yahoo"
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1d"
    auto_adjust: bool = False


@dataclass(frozen=True)
class EngineSpec:
    data: DataConfig
    indicators: IndicatorsConfig
    strategy: StrategyConfig
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    plot_indicators: List[str] = field(default_factory=list)
    periods_per_year: int = 252
    rf_annual: float = 0.0
    # Period filters (multi-window)
    include_windows: Optional[List[tuple[str, str]]] = None
    exclude_windows: Optional[List[tuple[str, str]]] = None




@dataclass(frozen=True)
class BacktestBundle:
    md: MarketData
    feats: FeaturesData
    signals: SignalFrame
    portfolio_result: PortfolioResult
    report: BacktestReport
    meta: Dict[str, Any] = field(default_factory=dict)


def _apply_time_windows_to_df(
    df: Any,
    include_windows: Optional[List[tuple[str, str]]],
    exclude_windows: Optional[List[tuple[str, str]]],
):
    import pandas as pd
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.sort_index()
    out = out[~out.index.isna()]

    idx = out.index
    idx_tz = getattr(idx, "tz", None)

    def _coerce_ts(x):
        ts = pd.to_datetime(x)
        if idx_tz is None:
            # index is tz-naive -> make boundary tz-naive
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_convert(None)
            return ts
        else:
            # index is tz-aware -> make boundary tz-aware in same tz
            if getattr(ts, "tzinfo", None) is None:
                ts = ts.tz_localize(idx_tz)
            else:
                ts = ts.tz_convert(idx_tz)
            return ts

    # include mask
    if include_windows:
        m_inc = pd.Series(False, index=idx)
        for s, e in include_windows:
            s_dt = _coerce_ts(s)
            e_dt = _coerce_ts(e)
            m_inc |= (idx >= s_dt) & (idx <= e_dt)

    else:
        m_inc = pd.Series(True, index=idx)

    # exclude mask
    if exclude_windows:
        m_exc = pd.Series(False, index=idx)
        for s, e in exclude_windows:
            s_dt = pd.to_datetime(s)
            e_dt = pd.to_datetime(e)
            m_exc |= (idx >= s_dt) & (idx <= e_dt)
    else:
        m_exc = pd.Series(False, index=idx)

    final_mask = m_inc & (~m_exc)
    return out.loc[final_mask.values]


def apply_period_filters(md: MarketData, cfg: DataConfig) -> MarketData:
    # apply multi-window slicing per symbol
    if (not cfg.include_windows) and (not cfg.exclude_windows):
        return md

    new_bars = {}
    for sym, bars in md.bars.items():
        new_bars[sym] = _apply_time_windows_to_df(bars, cfg.include_windows, cfg.exclude_windows)

    return MarketData(
        bars=new_bars,
        source=md.source,
        timezone=md.timezone,
        interval=md.interval,
    )


# -----------------------------
# Indicator spec inference helpers
# -----------------------------
def sma_specs_for_ma_cross(fast_window: int, slow_window: int) -> List[FeatureSpec]:
    """
    Ensures features are named exactly as your strategy expects:
      sma_{fast_window}, sma_{slow_window}
    """
    fast_window = int(fast_window)
    slow_window = int(slow_window)

    return [
        FeatureSpec(
            indicator="sma",
            params={"window": fast_window},
            inputs=("Close",),
            name=f"sma_{fast_window}",
            warmup=fast_window,
            output_mode="series",
        ),
        FeatureSpec(
            indicator="sma",
            params={"window": slow_window},
            inputs=("Close",),
            name=f"sma_{slow_window}",
            warmup=slow_window,
            output_mode="series",
        ),
    ]


def resolve_specs(ind_cfg: IndicatorsConfig, strat_cfg: StrategyConfig) -> List[FeatureSpec]:
    if ind_cfg.specs is not None:
        if not ind_cfg.specs:
            raise ValueError("IndicatorsConfig.specs is empty.")
        return ind_cfg.specs

    if ind_cfg.spec_builder is not None:
        specs = ind_cfg.spec_builder()
        if not specs:
            raise ValueError("IndicatorsConfig.spec_builder returned empty specs list.")
        return specs

    # Generic: build the strategy and ask it
    strat = build_strategy(strat_cfg.kind, strat_cfg.params)
    specs = strat.required_features()
    return list(specs) if specs is not None else []


# -----------------------------
# Strategy registry
# -----------------------------
def build_strategy(kind: str, params: Dict[str, Any]):
    k = (kind or "").lower()

    if k == "ma_cross":
        p = MovingAverageCrossParams(
            fast_window=int(params.get("sma_fast_window", params.get("fast_window", 20))),
            slow_window=int(params.get("sma_slow_window", params.get("slow_window", 50))),
            allow_short=bool(params.get("allow_short", False)),
            nan_policy=str(params.get("nan_policy", "flat")),
        )
        return MovingAverageCrossStrategy(p)

    if k == "sma_price":
        p = PriceAboveSMAParams(
            window=int(params.get("sma_window", params.get("window", 50))),
            allow_short=bool(params.get("allow_short", False)),
            nan_policy=str(params.get("nan_policy", "flat")),
        )
        return PriceAboveSMAStrategy(p)

    if k == "rsi":
        p = RSIParams(
            period=int(params.get("rsi_window", params.get("period", 14))),
            low=float(params.get("rsi_oversold", params.get("low", 30.0))),
            high=float(params.get("rsi_overbought", params.get("high", 70.0))),
            mode=str(params.get("mode", "reversal")),
            allow_short=bool(params.get("allow_short", False)),
            nan_policy=str(params.get("nan_policy", "flat")),
        )
        return RSIStrategy(p)

    if k == "macd":
        p = MACDParams(
            fast=int(params.get("macd_fast_window", params.get("fast", 12))),
            slow=int(params.get("macd_slow_window", params.get("slow", 26))),
            signal=int(params.get("macd_signal_window", params.get("signal", 9))),
            trigger=str(params.get("trigger", "cross")),
            allow_short=bool(params.get("allow_short", False)),
            nan_policy=str(params.get("nan_policy", "flat")),
        )
        return MACDStrategy(p)

    if k == "bollinger":
        p = BollingerParams(
            bb_window=int(params.get("bb_window", params.get("bbands_window", 20))),
            bb_k=float(params.get("bb_k", params.get("bbands_num_stddev", 2.0))),
            allow_short=bool(params.get("allow_short", False)),
            nan_policy=str(params.get("nan_policy", "flat")),
        )
        return BollingerBandsStrategy(p)
    
    if k == "buy_hold":
        p = BuyHoldParams(
            buy_pct_cash=float(params.get("buy_pct_cash", 1.0)),
            nan_policy=str(params.get("nan_policy", "flat")),
        )
        return BuyHoldStrategy(p)
    
    if k == "obv":
        p = OBVParams(
            obv_span=int(params.get("obv_span", params.get("span", 20))),
            allow_short=bool(params.get("allow_short", False)),
            nan_policy=str(params.get("nan_policy", "flat")),
        )
        return OBVStrategy(p)

    if k == "stoch_vwap":
        p = StochVWAPParams(
            k_window=int(params.get("k_window", 14)),
            d_window=int(params.get("d_window", 3)),
            smooth_k=int(params.get("smooth_k", 1)),
            vwap_window=int(params.get("vwap_window", 20)),
            allow_short=bool(params.get("allow_short", False)),
            nan_policy=str(params.get("nan_policy", "flat")),
        )
        return StochVWAPStrategy(p)

    if k == "ichimoku":
        p = IchimokuParams(
            tenkan=int(params.get("tenkan", 9)),
            kijun=int(params.get("kijun", 26)),
            senkou_b=int(params.get("senkou_b", 52)),
            shift=int(params.get("shift", 26)),
            allow_short=bool(params.get("allow_short", False)),
            nan_policy=str(params.get("nan_policy", "flat")),
        )
        return IchimokuStrategy(p)



    raise ValueError(f"Unknown strategy kind: {kind}")



# -----------------------------
# Data loaders
# -----------------------------
def load_marketdata_bmce(cfg: DataConfig) -> MarketData:
    if cfg.bmce_paths is None:
        raise ValueError("BMCE source selected but bmce_paths is None.")

    ds = BMCEDataSource(timezone=cfg.timezone)

    # BaseDataSource.load(...) in your project returns MarketData (not dict)
    md_or_bars = ds.load(
        symbols=cfg.symbols,
        start=cfg.start,
        end=cfg.end,
        interval=cfg.interval,
        paths=cfg.bmce_paths,
    )

    # ✅ Case 1: BaseDataSource returns MarketData (your current behavior)
    if isinstance(md_or_bars, MarketData):
        return md_or_bars

    # ✅ Case 2: if you ever change load() to return dict[str, DataFrame]
    if isinstance(md_or_bars, dict):
        for s in cfg.symbols:
            if s not in md_or_bars:
                raise KeyError(f"BMCEDataSource returned no bars for '{s}'. Keys: {list(md_or_bars.keys())}")
        return MarketData(
            bars=md_or_bars,
            source="BMCEDataSource",
            timezone=cfg.timezone,
            interval=cfg.interval,
        )

    raise TypeError(f"BMCEDataSource.load returned unsupported type: {type(md_or_bars)}")


def load_marketdata_yfinance(cfg: DataConfig) -> MarketData:
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("yfinance not installed. Add it to requirements or use BMCE source.") from e

    if cfg.interval != "1d":
        # keep explicit
        raise ValueError("yfinance loader in this engine currently supports interval='1d' only.")

    out: Dict[str, Any] = {}
    for sym in cfg.symbols:
        df = yf.download(
            tickers=sym,
            start=cfg.start,
            end=cfg.end,
            interval=cfg.yf_interval,
            auto_adjust=cfg.yf_auto_adjust,
            progress=False,
        )
        if df is None or len(df) == 0:
            raise ValueError(f"yfinance returned empty data for symbol '{sym}'.")

        required = {"Open", "High", "Low", "Close"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"yfinance data missing columns {missing}. got={list(df.columns)}")
        out[sym] = df.sort_index()

    return MarketData(bars=out, source="yfinance", timezone=cfg.timezone, interval=cfg.yf_interval)


def load_marketdata(cfg: DataConfig) -> MarketData:
    if cfg.source == "bmce":
        return load_marketdata_bmce(cfg)
    if cfg.source == "yfinance":
        return load_marketdata_yfinance(cfg)
    raise ValueError(f"Unknown data source: {cfg.source}")





def _coerce_boundary_to_index_tz(idx: pd.DatetimeIndex, ts: pd.Timestamp) -> pd.Timestamp:
    """Make boundary timestamp compatible with index timezone (tz-aware or tz-naive)."""
    idx_tz = getattr(idx, "tz", None)

    if idx_tz is None:
        # index tz-naive
        if getattr(ts, "tzinfo", None) is not None:
            return ts.tz_convert(None)
        return ts
    else:
        # index tz-aware
        if getattr(ts, "tzinfo", None) is None:
            return ts.tz_localize(idx_tz)
        return ts.tz_convert(idx_tz)


def _clean_dt_index(df: Any) -> Any:
    """Ensure df is sorted by a DatetimeIndex and has no NaT index."""
    if df is None or len(df) == 0:
        return df

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.sort_index()
    out = out[~out.index.isna()]
    return out


def _apply_windows_mask(
    df: Any,
    include_windows: Optional[List[Tuple[str, str]]] = None,
    exclude_windows: Optional[List[Tuple[str, str]]] = None,
) -> Any:
    """Apply include/exclude windows on a DatetimeIndex df."""
    if df is None or len(df) == 0:
        return df

    out = _clean_dt_index(df)
    idx = out.index

    # Include mask
    if include_windows:
        m_inc = pd.Series(False, index=idx)
        for s, e in include_windows:
            s_dt = _coerce_boundary_to_index_tz(idx, pd.to_datetime(s))
            e_dt = _coerce_boundary_to_index_tz(idx, pd.to_datetime(e))
            m_inc |= (idx >= s_dt) & (idx <= e_dt)
    else:
        m_inc = pd.Series(True, index=idx)

    # Exclude mask
    if exclude_windows:
        m_exc = pd.Series(False, index=idx)
        for s, e in exclude_windows:
            s_dt = _coerce_boundary_to_index_tz(idx, pd.to_datetime(s))
            e_dt = _coerce_boundary_to_index_tz(idx, pd.to_datetime(e))
            m_exc |= (idx >= s_dt) & (idx <= e_dt)
    else:
        m_exc = pd.Series(False, index=idx)

    return out.loc[(m_inc & (~m_exc)).values]


def slice_df_period(
    df: Any,
    start: Optional[str],
    end: Optional[str],
    include_windows: Optional[List[Tuple[str, str]]] = None,
    exclude_windows: Optional[List[Tuple[str, str]]] = None,
) -> Any:
    """
    Unified slicing:
    - first clean/sort index
    - then apply start/end slice (if provided)
    - then apply include/exclude windows (if provided)
    """
    if df is None or len(df) == 0:
        return df

    out = _clean_dt_index(df)
    idx = out.index

    if start:
        s = _coerce_boundary_to_index_tz(idx, pd.to_datetime(start))
        out = out.loc[out.index >= s]
        idx = out.index

    if end:
        e = _coerce_boundary_to_index_tz(idx, pd.to_datetime(end))
        out = out.loc[out.index <= e]

    out = _apply_windows_mask(out, include_windows=include_windows, exclude_windows=exclude_windows)
    return out


def slice_marketdata(
    md,
    start: Optional[str],
    end: Optional[str],
    include_windows: Optional[List[Tuple[str, str]]] = None,
    exclude_windows: Optional[List[Tuple[str, str]]] = None,
):
    """Slice MarketData.bars per symbol with start/end and include/exclude windows."""
    if md is None:
        return md

    new_bars: Dict[str, pd.DataFrame] = {}
    for sym, bars in md.bars.items():
        new_bars[sym] = slice_df_period(
            bars,
            start=start,
            end=end,
            include_windows=include_windows,
            exclude_windows=exclude_windows,
        )

    # Preserve the same MarketData type you already use
    return type(md)(
        bars=new_bars,
        source=getattr(md, "source", "unknown"),
        timezone=getattr(md, "timezone", "unknown"),
        interval=getattr(md, "interval", "unknown"),
        meta=getattr(md, "meta", {}),
    )


def slice_features(feats_full, md_sliced):
    """
    Align features index to exactly the sliced md bars index per symbol.
    This preserves indicator lookback effects because feats_full was computed on padded history,
    but you only *expose* the test window downstream.
    """
    if feats_full is None:
        return feats_full

    new_feats: Dict[str, pd.DataFrame] = {}
    for sym, bars in md_sliced.bars.items():
        f = feats_full.features.get(sym)
        if f is None:
            new_feats[sym] = pd.DataFrame(index=bars.index)
        else:
            f = _clean_dt_index(f)
            new_feats[sym] = f.reindex(bars.index)

    return type(feats_full)(
        features=new_feats,
        source=getattr(feats_full, "source", "unknown"),
        timezone=getattr(feats_full, "timezone", "unknown"),
        interval=getattr(feats_full, "interval", "unknown"),
        meta=getattr(feats_full, "meta", {}),
    )

# -----------------------------
# Engine
# -----------------------------
class BacktestEngine:
    def __init__(self, spec: EngineSpec) -> None:
        self.spec = spec

    def run(self) -> BacktestBundle:
        base_cfg = self.spec.data
        symbols = base_cfg.symbols

        specs = resolve_specs(self.spec.indicators, self.spec.strategy)
        max_warmup = int(max([getattr(s, "warmup", 0) or 0 for s in specs] + [0]))
        pad_bars = max_warmup * 2

        # ------------------------------------------------------------
        # 0) Compute "effective" trade window start/end
        #    If user did not specify start/end but did specify include_windows,
        #    infer them so padding works.
        # ------------------------------------------------------------
        eff_start = base_cfg.start
        eff_end = base_cfg.end

        inc = getattr(base_cfg, "include_windows", None) or []
        if (eff_start is None or eff_end is None) and inc:
            starts = [pd.to_datetime(s) for s, _ in inc]
            ends   = [pd.to_datetime(e) for _, e in inc]
            if eff_start is None:
                eff_start = min(starts).date().isoformat()
            if eff_end is None:
                eff_end = max(ends).date().isoformat()

        # ------------------------------------------------------------
        # 1) Build padded load config (based on effective start)
        # ------------------------------------------------------------
        load_cfg = base_cfg
        if eff_start and pad_bars > 0:
            start_dt = pd.to_datetime(eff_start)
            load_start_dt = start_dt - pd.tseries.offsets.BDay(pad_bars)
            load_cfg = replace(load_cfg, start=load_start_dt.date().isoformat())

        # Ensure we load enough to cover effective end too
        if eff_end:
            load_cfg = replace(load_cfg, end=eff_end)

        # 2) Load padded history
        md_loaded = load_marketdata(load_cfg)

        # ------------------------------------------------------------
        # 3) Slice ONLY by padded start/end for indicator computation
        #    IMPORTANT: do NOT apply include/exclude windows here.
        # ------------------------------------------------------------
        md_for_ind = slice_marketdata(
            md_loaded,
            start=load_cfg.start,
            end=load_cfg.end,
            include_windows=None,
            exclude_windows=None,
        )


        # 4) Compute indicators on continuous padded history
        if not specs:
            feats_full = FeaturesData(
                features={sym: pd.DataFrame(index=md_for_ind.bars[sym].index) for sym in symbols},
                source=getattr(md_for_ind, "source", "unknown"),
                timezone=getattr(md_for_ind, "timezone", "unknown"),
                interval=getattr(md_for_ind, "interval", "unknown"),
                meta={"specs": [], "engine_version": getattr(self.spec.indicators, "engine_version", "v1")},
            )
        else:
            ind = IndicatorEngine(
                cache_dir=self.spec.indicators.cache_dir,
                enable_disk_cache=self.spec.indicators.enable_disk_cache,
                enable_memory_cache=self.spec.indicators.enable_memory_cache,
                engine_version=self.spec.indicators.engine_version,
            )
            feats_full = ind.compute(md_for_ind, specs=specs, symbols=symbols)

        # ------------------------------------------------------------
        # 5) Slice TRUE backtest window (effective start/end + include/exclude)
        # ------------------------------------------------------------
        md = slice_marketdata(
            md_loaded,
            start=eff_start,
            end=eff_end,
            include_windows=getattr(base_cfg, "include_windows", None),
            exclude_windows=getattr(base_cfg, "exclude_windows", None),
        )


        # 5) Slice features to match md (so indicators keep lookback effects)
        feats = slice_features(feats_full, md)

        # 6) Strategy + Portfolio + Results
        strat = build_strategy(self.spec.strategy.kind, self.spec.strategy.params)
        sf = strat.generate_signals(md, feats, symbols=symbols)
        
        # --- MACD-only debug (guarded) ---
        if str(self.spec.strategy.kind).lower() == "macd":
            sym = symbols[0]
            sp = (self.spec.strategy.params or {})
            fast = int(sp.get("fast", sp.get("macd_fast_window", 12)))
            slow = int(sp.get("slow", sp.get("macd_slow_window", 26)))
            sig_win = int(sp.get("signal", sp.get("macd_signal_window", 9)))

            base = f"macd_{fast}_{slow}_{sig_win}"

            # safer access + better error message
            cols = feats.features[sym].columns
            if f"{base}__line" not in cols or f"{base}__signal" not in cols:
                raise KeyError(
                    f"Expected MACD columns not found: {base}__line / {base}__signal. "
                    f"Available macd cols: {[c for c in cols if c.startswith('macd_')][:30]}"
                )

            line = feats.features[sym][f"{base}__line"]
            sigl = feats.features[sym][f"{base}__signal"]
            hist = line - sigl





        port = PortfolioEngine(self.spec.portfolio)
        pres = port.run(md, sf, symbols=symbols)

        plot_inds = self.spec.plot_indicators or default_plot_indicators(
            self.spec.strategy.kind, self.spec.strategy.params
        )

        analyzer = ResultsAnalyzer(periods_per_year=self.spec.periods_per_year, rf_annual=self.spec.rf_annual)
        report = analyzer.analyze(
            pres,
            market_data=md,
            symbols=symbols,
            features_data=feats,
            plot_indicators=plot_inds,
            benchmark_market_data=None,
            benchmark_symbol=None,
        )

        return BacktestBundle(md=md, feats=feats, signals=sf, portfolio_result=pres, report=report, meta={})


def estimate_warmup_bars_from_params(spec: EngineSpec, active_params: list) -> int:
    """
    Conservative warmup bars needed BEFORE the trade_start so indicators are valid on day 1.
    Uses:
      - strategy kind + params (current)
      - optimizer search ranges if present in active_params
      - portfolio windows (ADV, volume gate) if enabled
    """
    k = (spec.strategy.kind or "").lower()
    p = spec.strategy.params or {}

    # --- strategy warmup ---
    warm = 0

    def _max_in_paramdef(name: str, default: int) -> int:
        # Try to find a ParamDef that matches this key and return its max candidate value.
        # You may need to adapt depending on your ParamDef schema.
        for pd_ in active_params:
            if getattr(pd_, "key", None) == name:
                # common patterns: pd_.values or (min,max,step)
                if hasattr(pd_, "values") and pd_.values:
                    return int(max(pd_.values))
                if hasattr(pd_, "max"):
                    return int(pd_.max)
        return int(default)

    if k == "ma_cross":
        # warmup is slow SMA window
        slow = int(p.get("sma_slow_window", p.get("slow_window", 50)))
        slow = _max_in_paramdef("strategy.sma_slow_window", slow)
        warm = max(warm, slow)

    elif k in ("sma_price", "price_sma", "price_above_sma"):
        w = int(p.get("sma_window", p.get("window", 50)))
        w = _max_in_paramdef("strategy.sma_window", w)
        warm = max(warm, w)

    elif k == "rsi":
        n = int(p.get("rsi_window", p.get("period", 14)))
        n = _max_in_paramdef("strategy.rsi_window", n)
        warm = max(warm, n)

    elif k == "macd":
        slow = int(p.get("macd_slow_window", p.get("slow", 26)))
        sig  = int(p.get("macd_signal_window", p.get("signal", 9)))
        slow = _max_in_paramdef("strategy.macd_slow_window", slow)
        sig  = _max_in_paramdef("strategy.macd_signal_window", sig)
        warm = max(warm, slow + sig)  # matches your MACDStrategy conservative warmup

    elif k == "bollinger":
        w = int(p.get("bb_window", 20))
        w = _max_in_paramdef("strategy.bb_window", w)
        warm = max(warm, w)

    elif k == "obv":
        span = int(p.get("obv_span", p.get("span", 20)))
        # +1 for diff/initialization; keep conservative
        warm = max(warm, span + 1)

    elif k == "stoch_vwap":
        kw = int(p.get("k_window", 14))
        dw = int(p.get("d_window", 3))
        sk = int(p.get("smooth_k", 1))
        vw = int(p.get("vwap_window", 20))
        warm = max(warm, max(kw, dw, sk, vw) + 1)

    elif k == "ichimoku":
        ten = int(p.get("tenkan", 9))
        kij = int(p.get("kijun", 26))
        sb  = int(p.get("senkou_b", 52))
        sh  = int(p.get("shift", 26))
        warm = max(warm, sb + sh)


    # --- portfolio warmup (ADV windows) ---
    port = spec.portfolio
    if getattr(port, "use_participation_cap", False) and str(getattr(port, "participation_basis", "")) == "adv":
        warm = max(warm, int(getattr(port, "adv_window", 1) or 1))
    if getattr(port, "use_volume_gate", False) and str(getattr(port, "volume_gate_kind", "")) == "min_ratio_adv":
        warm = max(warm, int(getattr(port, "volume_gate_adv_window", 1) or 1))

    # give a small cushion
    return int(warm + 5)

