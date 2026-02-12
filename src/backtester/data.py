"""
data.py — Research-first data layer with institutional-style separation:

- MarketData: normalized dataset container ("bundle-lite")
- BaseDataSource: stable interface (like QuantStart's DataHandler idea)
- YahooFinanceDataSource: yfinance adapter (testing)
- BMCEDataSource: CSV adapter for BMCE-like format

Canonical output schema per symbol:
Index: pd.DatetimeIndex (tz-aware, sorted, unique)
Columns: Open, High, Low, Close, Adj Close (optional), Volume
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import hashlib
import json

import pandas as pd


CANONICAL_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ----------------------------
# Layer 1: MarketData container
# ----------------------------
@dataclass(frozen=True)
class MarketData:
    """
    Normalized market data for research backtesting.

    bars: dict[symbol -> DataFrame] where each DF:
      - index: tz-aware DatetimeIndex, sorted ascending, unique
      - columns: Open, High, Low, Close, Volume (+ optional Adj Close)
    """
    bars: Dict[str, pd.DataFrame]
    source: str
    timezone: str = "GMT"
    interval: str = "1d"
    meta: Dict[str, object] = field(default_factory=dict)

    def symbols(self) -> List[str]:
        return list(self.bars.keys())

    def get(self, symbol: str) -> pd.DataFrame:
        return self.bars[symbol]


# ----------------------------
# Layer 4: Normalization utils
# ----------------------------
def _ensure_datetime_index(
    df: pd.DataFrame,
    tz: str = "GMT",
) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex after parsing.")
    # Sort, drop duplicates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Localize/convert timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    else:
        df.index = df.index.tz_convert(tz)

    return df


def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _standardize_ohlcv(
    df: pd.DataFrame,
    tz: str = "GMT",
    require_ohlc: bool = True,
    fill_adj_close: bool = True,
) -> pd.DataFrame:
    """
    Enforce canonical columns and a clean datetime index.
    """
    df = _ensure_datetime_index(df, tz=tz)

    # Normalize column names (common variants)
    rename_map = {
        "AdjClose": "Adj Close",
        "Adj_Close": "Adj Close",
        "adj_close": "Adj Close",
        "close": "Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Basic schema checks
    needed = ["Open", "High", "Low", "Close"] if require_ohlc else ["Close"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Adj Close handling
    if fill_adj_close and "Adj Close" not in df.columns:
        # For many research tasks, using Close is acceptable; keep explicit
        df["Adj Close"] = df["Close"]

    # Coerce numeric
    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = _coerce_numeric(df, numeric_cols)

    # Keep only canonical cols that exist (preserve order)
    keep = [c for c in CANONICAL_COLS if c in df.columns]
    df = df[keep]

    # Drop rows where close is missing
    df = df.dropna(subset=["Close"])

    return df


def slice_date_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz=df.index.tz)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end, tz=df.index.tz)]
    return df


def align_symbols(
    bars: Dict[str, pd.DataFrame],
    how: str = "inner",
    fill_method: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Align all symbols to a common index (inner = intersection; outer = union).
    Useful once you do multi-asset comparisons.

    fill_method: None, "ffill", "bfill"
    """
    if not bars:
        return bars

    indexes = [df.index for df in bars.values()]
    common_idx = indexes[0]
    for idx in indexes[1:]:
        common_idx = common_idx.intersection(idx) if how == "inner" else common_idx.union(idx)

    out: Dict[str, pd.DataFrame] = {}
    for sym, df in bars.items():
        tmp = df.reindex(common_idx)
        if fill_method is not None:
            tmp = tmp.fillna(method=fill_method)
        out[sym] = tmp
    return out


# ----------------------------
# Layer 2: BaseDataSource interface
# ----------------------------
class BaseDataSource:
    """
    Stable interface for data acquisition adapters.

    Inspired by the idea of a common DataHandler/Feed interface:
    - QuantStart: ABC interface to ensure compatibility across components :contentReference[oaicite:7]{index=7}
    - Backtrader: multiple data feeds that plug into the engine :contentReference[oaicite:8]{index=8}
    """

    def __init__(
        self,
        timezone: str = "GMT",
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
    ) -> None:
        self.timezone = timezone
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        symbols: Sequence[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        align: bool = False,
        align_how: str = "inner",
        fill_method: Optional[str] = None,
        **kwargs,
    ) -> MarketData:
        """
        Public API: return MarketData with normalized per-symbol OHLCV.

        align=True is helpful for multi-asset research comparisons.
        """
        symbols = list(symbols)
        cache_key = self._cache_key(symbols, start, end, interval, kwargs)

        if self.use_cache and self.cache_dir:
            cached = self._try_load_cache(cache_key)
            if cached is not None:
                bars = cached
            else:
                bars = self._load_impl(symbols, start, end, interval, **kwargs)
                self._save_cache(cache_key, bars)
        else:
            bars = self._load_impl(symbols, start, end, interval, **kwargs)

        # Normalize & slice
        normed: Dict[str, pd.DataFrame] = {}
        for sym, df in bars.items():
            df = _standardize_ohlcv(df, tz=self.timezone)
            df = slice_date_range(df, start, end)
            normed[sym] = df

        if align and len(normed) > 1:
            normed = align_symbols(normed, how=align_how, fill_method=fill_method)

        return MarketData(
            bars=normed,
            source=self.__class__.__name__,
            timezone=self.timezone,
            interval=interval,
            meta={
                "start": start,
                "end": end,
                "symbols": symbols,
                "interval": interval,
                **({"adapter_kwargs": kwargs} if kwargs else {}),
            },
        )

    # ---- adapter-specific implementation hook
    def _load_impl(
        self,
        symbols: Sequence[str],
        start: Optional[str],
        end: Optional[str],
        interval: str,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    # ---- caching (simple, research-friendly)
    def _cache_key(
        self,
        symbols: Sequence[str],
        start: Optional[str],
        end: Optional[str],
        interval: str,
        kwargs: dict,
    ) -> str:
        payload = {
            "cls": self.__class__.__name__,
            "symbols": list(symbols),
            "start": start,
            "end": end,
            "interval": interval,
            "timezone": self.timezone,
            "kwargs": kwargs,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _try_load_cache(self, key: str) -> Optional[Dict[str, pd.DataFrame]]:
        assert self.cache_dir is not None
        path = self.cache_dir / f"{key}.pkl"
        if not path.exists():
            return None
        return pd.read_pickle(path)

    def _save_cache(self, key: str, bars: Dict[str, pd.DataFrame]) -> None:
        assert self.cache_dir is not None
        path = self.cache_dir / f"{key}.pkl"
        pd.to_pickle(bars, path)


# ----------------------------
# Layer 3a: yfinance adapter (testing)
# ----------------------------
class YahooFinanceDataSource(BaseDataSource):
    """
    Research-friendly Yahoo adapter (yfinance).

    Notes:
    - yfinance often returns MultiIndex columns if multiple tickers are requested.
    - We convert to dict[symbol -> DataFrame] to keep downstream logic explicit.
    """

    def _load_impl(
        self,
        symbols: Sequence[str],
        start: Optional[str],
        end: Optional[str],
        interval: str,
        auto_adjust: bool = False,
        progress: bool = False,
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("yfinance is not installed. `pip install yfinance`") from e

        # yfinance download
        df = yf.download(
            tickers=list(symbols),
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=progress,
            group_by="column",
            **kwargs,
        )

        out: Dict[str, pd.DataFrame] = {}

        # Multi-ticker format: columns are MultiIndex: (field, ticker)
        if isinstance(df.columns, pd.MultiIndex):
            # fields: Open/High/Low/Close/Adj Close/Volume
            for sym in symbols:
                sub = df.xs(sym, axis=1, level=1, drop_level=True).copy()
                out[sym] = sub
        else:
            # Single ticker: columns are normal
            sym = symbols[0] if symbols else "UNKNOWN"
            out[sym] = df.copy()

        # Index is usually naive GMT; normalization will localize to self.timezone
        return out


# ----------------------------
# Layer 3b: BMCE-like CSV adapter
# Example CSV:
# Date,"Price","Open","High","Low","Vol.","Change %"
# 01/23/2025,"90.90","89.51","90.90","89.26","355.87K","1.91%"
# ----------------------------
def _parse_human_volume(x: object) -> float:
    """
    Convert strings like '355.87K', '1.2M', '3B' into numeric volume.
    Returns float to preserve fidelity; you can cast to int later if desired.
    """
    if pd.isna(x):
        return float("nan")
    s = str(x).strip().replace(",", "")
    if s == "":
        return float("nan")

    mult = 1.0
    last = s[-1].upper()
    if last in ("K", "M", "B"):
        s_num = s[:-1]
        mult = {"K": 1e3, "M": 1e6, "B": 1e9}[last]
    else:
        s_num = s

    try:
        return float(s_num) * mult
    except ValueError:
        return float("nan")


def _parse_percent(x: object) -> float:
    """'1.91%' -> 0.0191"""
    if pd.isna(x):
        return float("nan")
    s = str(x).strip().replace("%", "")
    if s == "":
        return float("nan")
    try:
        return float(s) / 100.0
    except ValueError:
        return float("nan")


class BMCEDataSource(BaseDataSource):
    """
    CSV adapter for the given BMCE-like format.

    Design is intentionally "Generic CSV"-style: map input fields to canonical OHLCV,
    similar in spirit to Backtrader's Generic CSV support. :contentReference[oaicite:9]{index=9}
    """

    def __init__(
        self,
        timezone: str = "GMT",
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
        dayfirst: bool = False,
        date_format: Optional[str] = None,
    ) -> None:
        super().__init__(timezone=timezone, cache_dir=cache_dir, use_cache=use_cache)
        self.dayfirst = dayfirst
        self.date_format = date_format

    def _load_impl(
        self,
        symbols: Sequence[str],
        start: Optional[str],
        end: Optional[str],
        interval: str,
        paths: Union[str, Path, Dict[str, Union[str, Path]]],
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """
        paths:
        - if single symbol: a single path
        - if multiple symbols: dict {symbol: path}
        """
        if interval != "1d":
            # You can extend later; keep explicit now
            raise ValueError("BMCEDataSource currently supports daily bars only (interval='1d').")

        out: Dict[str, pd.DataFrame] = {}

        if isinstance(paths, (str, Path)):
            p = Path(paths)
            ext = p.suffix.lower()

            # ✅ Special case: ONE workbook contains MANY symbols as sheets
            if ext in [".xlsx", ".xls"] and len(symbols) >= 1:
                # Open once (faster than reopening per symbol)
                xls = pd.ExcelFile(p)
                available = set(xls.sheet_names)

                for sym in symbols:
                    if sym not in available:
                        raise ValueError(
                            f"Workbook '{p.name}' has no sheet '{sym}'. "
                            f"Available sheets: {xls.sheet_names}"
                        )
                    out[sym] = self._read_one_file(p, sheet_name=sym)

            else:
                # ✅ Original behavior (single file = single symbol)
                if len(symbols) != 1:
                    raise ValueError(
                        "If `paths` is a single path, `symbols` must have length 1 "
                        "(unless it's an .xlsx workbook with one sheet per symbol)."
                    )
                sym = symbols[0]
                out[sym] = self._read_one_file(p)

        else:
            # dict mapping (already works)
            for sym in symbols:
                if sym not in paths:
                    raise ValueError(f"Missing path for symbol '{sym}'. Provided keys: {list(paths.keys())}")
                out[sym] = self._read_one_file(Path(paths[sym]))

        return out


    def _read_one_file(self, path: Path, sheet_name: str | None = None) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # 1) Read file depending on extension
        ext = path.suffix.lower()
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        elif ext == ".csv":
            df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # 2) Clean column names (BMCE exports often have spaces)
        df.columns = df.columns.astype(str).str.strip()

        # 3) Rename BMCE columns -> canonical raw names
        rename = {
            "Ouvt": "Open",
            "'+Haut": "High",
            "'+Bas": "Low",
            "Clôture": "Close",
            "Volume": "Volume",
        }
        df = df.rename(columns=rename)

        # 4) Parse Date and set index
        if "Date" not in df.columns:
            raise ValueError(f"'Date' column not found. Found columns: {list(df.columns)}")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # add dayfirst=True if needed
        df = df.dropna(subset=["Date"]).set_index("Date")

        # 5) Convert numbers (handle comma decimals if they appear)
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns and df[c].dtype == "object":
                df[c] = (
                    df[c].astype(str)
                    .str.replace(" ", "", regex=False)
                    .str.replace(",", ".", regex=False)
                )
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # 6) Keep only OHLCV
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        missing = [c for c in ["Open", "High", "Low", "Close"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLC columns after rename: {missing}. Columns: {list(df.columns)}")

        return df[keep]


    