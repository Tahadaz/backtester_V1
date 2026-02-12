from __future__ import annotations

import streamlit as st
st.set_page_config(page_title="Backtester", layout="wide")

from plotly.subplots import make_subplots
import copy
import io
import zipfile
from datetime import datetime

import os
import tempfile
from pathlib import Path
from dataclasses import replace
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from optimize import build_spec_from_result_row, batch_optimize_by_period


import plotly.graph_objects as go

# ---- Project imports ----
from engine import BacktestEngine, BenchmarkConfig, EngineSpec, DataConfig, IndicatorsConfig, StrategyConfig
from portfolio import PortfolioConfig, CostModel
from optimize import (
    OptimizeConfig,
    TrialResult,
    ParamDef,
    default_param_catalog,
    run_optimization,
)
from data import BMCEDataSource, YahooFinanceDataSource
from results import ResultsAnalyzer
from site_publish import export_run_to_results_site


# -----------------------------
# Period catalogs (edit freely)
# -----------------------------
MASI_PERIODS: dict[str, tuple[str, str]] = {
    "Pre-2008 (2005-2007)": ("2005-01-01", "2007-12-31"),
    "Crise (2008-2009)": ("2008-01-01", "2009-12-31"),
    "Mid-crisis recovery (2010)": ("2010-01-01", "2010-12-31"),
    "Primtemps arabe / Eurozone (2011-2013)": ("2011-01-01", "2013-12-31"),
    "Normalization (2014-2019)": ("2014-01-01", "2019-12-31"),
    "COVID shock (2020)": ("2020-01-01", "2020-12-31"),
    "Inflation / rates shock (2022-mid2023)": ("2022-01-01", "2023-06-01"),
    "Post-2024 (2024-06-25+)": ("2023-06-02", "2025-12-31"),
}

IAM_PERIODS: dict[str, tuple[str, str]] = {
   "Etisalat control transition (2013-2015)": ("2013-01-01", "2015-12-31"),
    "Affaire Inwi (2024-01-29 to 2025-03-01)": ("2024-01-29", "2025-03-01"),
    "Changement Leadership (2025-03-01+)": ("2025-03-01", "2026-12-31"),
}
# =========================
# Manual-list default values
# =========================
# Put this near your helpers (e.g., right above parse_int_list), so BOTH editors can use it.
from run_spec import build_run_spec, run_id_from_spec
from site_export import export_store_to_run_folder

RESULTS_ROOT = Path("results_website/assets/results")  # relative to repo root

RESULTS_ROOT = (Path(__file__).parent / "results_website" / "assets" / "results").resolve()
RUNS_DIR = RESULTS_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure index.json exists (minimal)
idx = RESULTS_ROOT / "index.json"
if not idx.exists():
    idx.write_text('{"latest": null, "runs": []}', encoding="utf-8")

MANUAL_DEFAULTS: dict[str, str] = {
    # --- SMA price ---
    "strategy.sma_window": "5,10,14,20,30,50,100,200",

    # --- MA cross ---
    # fast < slow is enforced elsewhere; these are just suggested sets
    "strategy.sma_fast_window": "5,8,10,12,15,20,30",
    "strategy.sma_slow_window": "20,30,50,80,100,150,200",

    # --- RSI ---
    "strategy.rsi_window": "7,10,14,21,28",
    "strategy.rsi_oversold": "20,25,30,35",
    "strategy.rsi_overbought": "65,70,75,80",

    # --- MACD ---
    "strategy.macd_fast_window": "8,10,12,15,26",
    "strategy.macd_slow_window": "20,26,30,36,50",
    "strategy.macd_signal_window": "5,7,9,12,20",

    # --- Bollinger ---
    "strategy.bb_window": "10,14,20,30,50",
    "strategy.bb_k": "1,1.5,2,2.5,3",

    # --- Cooldown / sizing (if you expose them in the catalog) ---
    "portfolio.cooldown_bars": "0,5,10,21,60,120",
    "portfolio.buy_pct_cash": "0.1,0.25,0.5,0.75,1.0",
    "portfolio.sell_pct_shares": "0.1,0.25,0.5,0.75,1.0",
    "portfolio.min_return_before_sell": "0.0,0.01,0.02,0.05,0.10",
}
import json

def format_params_for_table(params: dict) -> str:
    # compact, stable order, easy to read
    return json.dumps(params or {}, sort_keys=True)
import io, json, zipfile
from datetime import datetime

def _ensure_site_store():
    if "site_export_store" not in st.session_state:
        # maps ticker -> payload
        st.session_state["site_export_store"] = {}
def _run_folder_has_site(run_dir: Path) -> bool:
    # Minimum sanity: manifest + at least one html plot somewhere
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        return False
    htmls = list(run_dir.glob("stocks/*/plots/*.html"))
    return len(htmls) > 0
def _extract_zip_bytes_to_folder(zip_bytes: bytes, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        z.extractall(out_dir)

def _push_ticker_results_to_site_store(
    *,
    ticker: str,
    leaderboard_df: pd.DataFrame,
    best_bundles: dict[str, Any],
    best_specs: dict[str, EngineSpec],
    top5_names: list[str],
    title: str = "Backtest Results",
):
    """
    Save/overwrite ONE ticker’s optimized leaderboard+plots in session_state.
    This lets you run multiple tickers and later download one site ZIP with all.
    """
    _ensure_site_store()

    # Normalize kinds from "ma_cross (best)" => "ma_cross"
    top5_kinds = [n.replace(" (best)", "").strip() for n in top5_names]

    plots_map = {}
    ledgers_map = {}

    for kind in top5_kinds:
        if kind not in best_bundles or kind not in best_specs:
            continue

        b = best_bundles[kind]
        spec = best_specs[kind]

        sym0 = b.md.symbols()[0]
        bars = b.md.bars[sym0]
        pp = b.report.plots.get("price_panel", {})

        fig = plot_price_indicators_trades_line(
            bars=bars,
            strategy_params=spec.strategy.params,
            indicators=pp.get("indicators"),
            trades=pp.get("trades"),
            indicator_cols=pp.get("indicator_cols"),
            port_cfg=spec.portfolio,
        )

        # HTML plot
        plots_map[f"{kind}_price_panel.html"] = fig.to_html(full_html=True, include_plotlyjs="cdn")

        # Trade ledger
        tables = getattr(b, "report", None).tables if getattr(b, "report", None) is not None else {}
        tl = tables.get("trade_ledger", None)
        if isinstance(tl, pd.DataFrame) and not tl.empty:
            ledgers_map[f"{kind}_trade_ledger.csv"] = tl.to_csv(index=False)
        else:
            ledgers_map[f"{kind}_trade_ledger_EMPTY.txt"] = "No trade_ledger table found or it is empty.\n"


    st.session_state["site_export_store"][ticker] = {
        "site_title": title,
        "leaderboard_df": leaderboard_df.copy(),
        "best_bundles": best_bundles,   # required by zip builder
        "best_specs": best_specs,       # likely required
        "top5_kinds": top5_kinds, 
        "plots": plots_map,
        "ledgers": ledgers_map,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }

def latest_signal_from_kind(
    *,
    kind: str,
    bars: pd.DataFrame,
    indicators: pd.DataFrame | None,
    params: dict | None,
) -> tuple[pd.Timestamp, int, str]:
    """
    Returns (signal_date, signal_today, signal_label) using the strategy rules
    on the LAST available bar.
      signal_today: +1 BUY, -1 SELL, 0 HOLD
    """
    kind = (kind or "").lower()
    p = params or {}
    df = bars.sort_index().copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    signal_date = pd.Timestamp(df.index[-1])
    close = float(pd.to_numeric(df["Close"], errors="coerce").iloc[-1])

    ind = None
    if indicators is not None and not indicators.empty:
        ind = indicators.copy()
        if not isinstance(ind.index, pd.DatetimeIndex):
            ind.index = pd.to_datetime(ind.index, errors="coerce")
        ind = ind.reindex(df.index)

    def _label(s: int) -> str:
        return "BUY" if s > 0 else "SELL" if s < 0 else "HOLD"

    # ---------------- SMA price ----------------
    if kind == "sma_price":
        w = int(p.get("sma_window", 50))
        sma = None
        if ind is not None and f"sma_{w}" in ind.columns:
            sma = float(pd.to_numeric(ind[f"sma_{w}"], errors="coerce").iloc[-1])
        else:
            sma = float(pd.to_numeric(df["Close"], errors="coerce").rolling(w, min_periods=w).mean().iloc[-1])
        if not np.isfinite(sma):
            return signal_date, 0, "NA"
        s = 1 if close > sma else -1
        return signal_date, s, _label(s)

    # ---------------- MA cross ----------------
    if kind == "ma_cross":
        f = int(p.get("sma_fast_window", 20))
        s_ = int(p.get("sma_slow_window", 50))
        fast = slow = None
        if ind is not None and f"sma_{f}" in ind.columns and f"sma_{s_}" in ind.columns:
            fast = float(pd.to_numeric(ind[f"sma_{f}"], errors="coerce").iloc[-1])
            slow = float(pd.to_numeric(ind[f"sma_{s_}"], errors="coerce").iloc[-1])
        else:
            c = pd.to_numeric(df["Close"], errors="coerce")
            fast = float(c.rolling(f, min_periods=f).mean().iloc[-1])
            slow = float(c.rolling(s_, min_periods=s_).mean().iloc[-1])
        if not (np.isfinite(fast) and np.isfinite(slow)):
            return signal_date, 0, "NA"
        sig = 1 if fast > slow else -1
        return signal_date, sig, _label(sig)

    # ---------------- RSI ----------------
    if kind == "rsi":
        w = int(p.get("rsi_window", 14))
        lo = float(p.get("rsi_oversold", 30))
        hi = float(p.get("rsi_overbought", 70))
        mode = str(p.get("mode", "reversal")).lower()

        rsi_val = None
        if ind is not None and f"rsi_{w}" in ind.columns:
            rsi_val = float(pd.to_numeric(ind[f"rsi_{w}"], errors="coerce").iloc[-1])
        else:
            # fallback: compute RSI quickly
            c = pd.to_numeric(df["Close"], errors="coerce")
            delta = c.diff()
            up = delta.clip(lower=0).rolling(w, min_periods=w).mean()
            dn = (-delta.clip(upper=0)).rolling(w, min_periods=w).mean()
            rs = up / dn.replace(0, np.nan)
            rsi_val = float((100 - (100 / (1 + rs))).iloc[-1])

        if not np.isfinite(rsi_val):
            return signal_date, 0, "NA"

        if mode == "momentum":
            # momentum: buy strength, sell weakness
            sig = 1 if rsi_val >= hi else -1 if rsi_val <= lo else 0
        else:
            # reversal (default): buy oversold, sell overbought
            sig = 1 if rsi_val <= lo else -1 if rsi_val >= hi else 0

        return signal_date, int(sig), _label(int(sig))

    # ---------------- MACD ----------------
    if kind == "macd":
        fast = int(p.get("macd_fast_window", 12))
        slow = int(p.get("macd_slow_window", 26))
        sigw = int(p.get("macd_signal_window", 9))
        trigger = str(p.get("trigger", "cross")).lower()

        base = f"macd_{fast}_{slow}_{sigw}"
        macd_line = macd_sig = None

        if ind is not None and f"{base}__line" in ind.columns and f"{base}__signal" in ind.columns:
            macd_line = float(pd.to_numeric(ind[f"{base}__line"], errors="coerce").iloc[-1])
            macd_sig = float(pd.to_numeric(ind[f"{base}__signal"], errors="coerce").iloc[-1])
        else:
            # fallback compute
            c = pd.to_numeric(df["Close"], errors="coerce")
            ema_f = c.ewm(span=fast, adjust=False, min_periods=fast).mean()
            ema_s = c.ewm(span=slow, adjust=False, min_periods=slow).mean()
            line = ema_f - ema_s
            sigl = line.ewm(span=sigw, adjust=False, min_periods=sigw).mean()
            macd_line = float(line.iloc[-1])
            macd_sig = float(sigl.iloc[-1])

        if not (np.isfinite(macd_line) and np.isfinite(macd_sig)):
            return signal_date, 0, "NA"

        if trigger == "zero":
            sig = 1 if macd_line > 0 else -1
        else:
            sig = 1 if macd_line > macd_sig else -1

        return signal_date, int(sig), _label(int(sig))

    # ---------------- Bollinger ----------------
    if kind == "bollinger":
        w = int(p.get("bb_window", 20))
        k = float(p.get("bb_k", 2.0))

        mid = std = None
        if ind is not None and f"sma_{w}" in ind.columns:
            mid = float(pd.to_numeric(ind[f"sma_{w}"], errors="coerce").iloc[-1])
        else:
            mid = float(pd.to_numeric(df["Close"], errors="coerce").rolling(w, min_periods=w).mean().iloc[-1])

        if ind is not None and f"std_{w}" in ind.columns:
            std = float(pd.to_numeric(ind[f"std_{w}"], errors="coerce").iloc[-1])
        else:
            std = float(pd.to_numeric(df["Close"], errors="coerce").rolling(w, min_periods=w).std().iloc[-1])

        if not (np.isfinite(mid) and np.isfinite(std)):
            return signal_date, 0, "NA"

        upper = mid + k * std
        lower = mid - k * std

        sig = 1 if close < lower else -1 if close > upper else 0
        return signal_date, int(sig), _label(int(sig))

    return signal_date, 0, "NA"
def run_opt_leaderboard_for_one_symbol(
    *,
    symbol_i: str,
    base_spec: EngineSpec,
    lb_opt_kinds: list[str],
    opt_cfg: OptimizeConfig,
    rank_metric: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, EngineSpec], list[str]]:

    rows = []
    best_specs: dict[str, EngineSpec] = {}
    best_bundles: dict[str, Any] = {}

    base_spec_i = replace(base_spec, data=replace(base_spec.data, symbols=[symbol_i]))

    base_spec_common = replace(
        base_spec_i,
        strategy=StrategyConfig(kind="buy_hold", params={"buy_pct_cash": 1.0}),
        benchmark=BenchmarkConfig(enabled=False),
        plot_indicators=[],
    )

    # 1) Optional Buy&Hold row
    if "buy_hold" in lb_opt_kinds:
        bh_key = ("bundle", _spec_key(base_spec_common))
        bh_bundle = st.session_state.get(bh_key)
        if bh_bundle is None:
            bh_bundle = BacktestEngine(base_spec_common).run()
            st.session_state[bh_key] = bh_bundle

        best_specs["buy_hold"] = base_spec_common
        best_bundles["buy_hold"] = bh_bundle

        row = leaderboard_row_from_report(bh_bundle.report, name="buy_hold")

        # ✅ add a signal column for consistency (buy_hold has no real “signal”)
        sym0 = bh_bundle.md.symbols()[0]
        bars_bh = bh_bundle.md.bars[sym0]
        sig_date = pd.Timestamp(bars_bh.index[-1]).date().isoformat() if len(bars_bh.index) else None
        row["Signal Date"] = sig_date
        row["Signal"] = "HOLD"
        row["Signal (-1/0/+1)"] = 0

        row["Best Params"] = format_params_for_table(base_spec_common.strategy.params)
        rows.append(row)

    # 2) Optimize each selected strategy kind
    for k in lb_opt_kinds:
        if k == "buy_hold":
            continue

        active_params_k = build_active_params_for_kind(k)
        if not active_params_k:
            continue

        base_spec_k = replace(
            base_spec_common,
            strategy=StrategyConfig(kind=k, params={}),  # let optimizer fill
            benchmark=BenchmarkConfig(enabled=False),
            plot_indicators=[],
        )

        best_k, top_df_k, best_params_k, best_spec_k, ranked_df_k = run_optimization(
            base_spec=base_spec_k,
            active_params=active_params_k,
            cfg=opt_cfg,
        )

        key_best = ("bundle", _spec_key(best_spec_k))
        bundle_k = st.session_state.get(key_best)
        if bundle_k is None:
            bundle_k = BacktestEngine(best_spec_k).run()
            st.session_state[key_best] = bundle_k

        best_specs[k] = best_spec_k
        best_bundles[k] = bundle_k

        row = leaderboard_row_from_report(bundle_k.report, name=f"{k} (best)")

        # ✅ ADD SIGNALS (this is what your all-sheets ZIP was missing)
        sym0 = bundle_k.md.symbols()[0]
        bars_k = bundle_k.md.bars[sym0]
        pp_k = bundle_k.report.plots.get("price_panel", {})
        ind_k = pp_k.get("indicators")

        sig_date, sig_num, sig_lab = latest_signal_from_kind(
            kind=k,
            bars=bars_k,
            indicators=ind_k,
            params=best_spec_k.strategy.params,
        )

        row["Signal Date"] = sig_date.date().isoformat() if sig_date is not None else None
        row["Signal"] = sig_lab
        row["Signal (-1/0/+1)"] = int(sig_num)

        row["Best Params"] = format_params_for_table(best_params_snapshot(best_spec_k))
        rows.append(row)

    df_lb = pd.DataFrame(rows)
    if df_lb.empty:
        return df_lb, best_bundles, best_specs, []

    df_lb = df_lb.sort_values(rank_metric, ascending=False, na_position="last").reset_index(drop=True)
    top5_names = df_lb.head(5)["Strategy"].tolist()
    return df_lb, best_bundles, best_specs, top5_names


def _make_site_zip_from_store(
    *,
    store: dict,
    site_title: str = "Backtest Results",
    about_subtitle: str = "Technical-analysis strategy backtests and parameter optimization",
    about_bullets: list[str] | None = None,
) -> bytes:
    """
    Build a website-ready ZIP that the standalone Streamlit site can read.

    ZIP layout:
      manifest.json
      stocks/<TICKER>/leaderboard.csv
      stocks/<TICKER>/plots/<strategy_key>_price_panel.html (+ svg/png if available)
      stocks/<TICKER>/profile.json  (placeholder)
      stocks/<TICKER>/notes.json    (placeholder)
    """
    if about_bullets is None:
        about_bullets = [
            "We test multiple TA strategies (SMA, MA Cross, RSI, MACD, Bollinger).",
            "We optimize parameters per strategy and compare performance.",
            "We report risk and trading statistics and visualize signals."
        ]

    buf = io.BytesIO()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare manifest
    tickers = sorted(store.keys())
    manifest = {
        "site_title": site_title,
        "about": {"subtitle": about_subtitle, "bullets": about_bullets},
        "stocks": [{"ticker": t, "name": t} for t in tickers],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))

        for ticker in tickers:
            payload = store[ticker]
            df_lb = payload["leaderboard_df"]
            best_bundles = payload["best_bundles"]
            best_specs = payload["best_specs"]
            top5_kinds = payload["top5_kinds"]

            # 1) leaderboard
            z.writestr(f"stocks/{ticker}/leaderboard.csv", df_lb.to_csv(index=False).encode("utf-8"))

            # 2) placeholders for stock presentation + explanations (you can customize later)
            profile = {
                "ticker": ticker,
                "name": ticker,
                "sector": "",
                "exchange": "",
                "currency": "",
                "summary": "",
                "key_points": []
            }
            notes = {
                "global": "Results compare multiple strategies on the same data range and execution assumptions.",
                "strategies": {}
            }
            z.writestr(f"stocks/{ticker}/profile.json", json.dumps(profile, indent=2).encode("utf-8"))
            z.writestr(f"stocks/{ticker}/notes.json", json.dumps(notes, indent=2).encode("utf-8"))

            # 3) plots
            for kind in top5_kinds:
                if kind not in best_bundles or kind not in best_specs:
                    continue
                b = best_bundles[kind]
                spec = best_specs[kind]

                sym0 = b.md.symbols()[0]
                bars = b.md.bars[sym0]
                pp = b.report.plots.get("price_panel", {})

                fig = plot_price_indicators_trades_line(
                    bars=bars,
                    strategy_params=spec.strategy.params,
                    indicators=pp.get("indicators"),
                    trades=pp.get("trades"),
                    indicator_cols=pp.get("indicator_cols"),
                    port_cfg=spec.portfolio,
                )

                # Always include HTML (best for "website")
                html = fig.to_html(full_html=True, include_plotlyjs="cdn")
                rel = f"{kind}_price_panel.html"
                z.writestr(f"stocks/{ticker}/plots/{rel}", html.encode("utf-8"))


                # 4) closed-trade ledger (PnL per closed trade)
                tables = getattr(b, "report", None).tables if getattr(b, "report", None) is not None else {}
                tl = tables.get("trade_ledger", None)

                if isinstance(tl, pd.DataFrame) and not tl.empty:
                    z.writestr(
                        f"stocks/{ticker}/ledgers/{kind}_trade_ledger.csv",
                        tl.to_csv(index=False).encode("utf-8"),
                    )
                else:
                    z.writestr(
                        f"stocks/{ticker}/ledgers/{kind}_trade_ledger_EMPTY.txt",
                        b"No trade_ledger table found or it is empty.",
                    )


                

        z.writestr("README.txt", f"Generated: {datetime.now().isoformat(timespec='seconds')}\n".encode("utf-8"))

    buf.seek(0)
    return buf.getvalue()

def default_manual_txt_for_param(param_key: str, pdef) -> str:
    """
    Returns the default text shown in the manual-list input for a given ParamDef key.
    - If MANUAL_DEFAULTS has an entry, use it.
    - Else fall back to a simple [lo, mid, hi] built from pdef.domain when possible.
    """
    if param_key in MANUAL_DEFAULTS:
        return MANUAL_DEFAULTS[param_key]

    # fallback: use domain if it's range-like
    try:
        lo, hi, step = pdef.domain
        # handle int vs float
        if pdef.kind == "int":
            mid = int((int(lo) + int(hi)) // 2)
            return f"{int(lo)},{mid},{int(hi)}"
        if pdef.kind == "float":
            mid = (float(lo) + float(hi)) / 2.0
            return f"{float(lo):g},{mid:g},{float(hi):g}"
    except Exception:
        pass

    return ""


import optimize as _opt_mod
st.title("Backtester (TA) — Backtest / Optimize")
st.sidebar.caption(f"optimize.py loaded from: {_opt_mod.__file__}")
st.sidebar.caption(f"run_optimization: {_opt_mod.run_optimization.__module__}.{_opt_mod.run_optimization.__name__}")
# ============================================================
# Upload persistence (FIXES caching + speed)
#   - Streamlit reruns were writing uploads to a NEW temp path each run,
#     which breaks caching keys and forces full recompute.
#   - We persist uploads to a deterministic path based on file content hash.
# ============================================================

from dataclasses import replace
def render_strategy_params_ui(kind: str, *, prefix: str, allow_short: bool, nan_policy: str = "flat") -> dict:
    k = (kind or "").lower()
    p = {}

    if k == "buy_hold":
        p["buy_pct_cash"] = st.slider(f"{prefix} buy_pct_cash", 0.01, 1.00, 1.0, 0.01, key=f"{prefix}_bh_buy")
        p["nan_policy"] = nan_policy
        return p

    if k == "ma_cross":
        fast = st.number_input(f"{prefix} fast_window", 2, 500, 20, 1, key=f"{prefix}_fast")
        slow = st.number_input(f"{prefix} slow_window", 3, 500, 50, 1, key=f"{prefix}_slow")
        if fast >= slow:
            st.warning("fast must be < slow; auto-adjusting")
            fast = min(int(fast), int(slow) - 1)
        p.update({"sma_fast_window": int(fast), "sma_slow_window": int(slow)})

    elif k == "sma_price":
        w = st.number_input(f"{prefix} sma_window", 2, 500, 50, 1, key=f"{prefix}_w")
        p["sma_window"] = int(w)

    elif k == "rsi":
        w = st.number_input(f"{prefix} rsi_window", 2, 500, 14, 1, key=f"{prefix}_rsiw")
        lo = st.number_input(f"{prefix} oversold", 1, 49, 30, 1, key=f"{prefix}_rsilo")
        hi = st.number_input(f"{prefix} overbought", 51, 99, 70, 1, key=f"{prefix}_rsihi")
        if lo >= hi:
            st.warning("oversold must be < overbought; auto-adjusting")
            lo = min(int(lo), int(hi) - 1)
        p.update({"rsi_window": int(w), "rsi_oversold": float(lo), "rsi_overbought": float(hi)})

    elif k == "macd":
        fast = st.number_input(f"{prefix} macd_fast", 2, 500, 12, 1, key=f"{prefix}_macdf")
        slow = st.number_input(f"{prefix} macd_slow", 3, 500, 26, 1, key=f"{prefix}_macds")
        sig  = st.number_input(f"{prefix} macd_signal", 2, 500, 9, 1, key=f"{prefix}_macdsg")
        if fast >= slow:
            st.warning("fast must be < slow; auto-adjusting")
            fast = min(int(fast), int(slow) - 1)
        p.update({"macd_fast_window": int(fast), "macd_slow_window": int(slow), "macd_signal_window": int(sig)})

    elif k == "bollinger":
        w = st.number_input(f"{prefix} bb_window", 2, 500, 20, 1, key=f"{prefix}_bbw")
        kk = st.number_input(f"{prefix} bb_k", 0.1, 5.0, 2.0, 0.1, key=f"{prefix}_bbk")
        p.update({"bb_window": int(w), "bb_k": float(kk)})

    # common
    p["allow_short"] = bool(allow_short)
    p["nan_policy"] = nan_policy
    return p


import json
def _make_leaderboard_zip(
    *,
    leaderboard_df: pd.DataFrame,
    top5_kinds: list[str],
    best_bundles: dict[str, Any],
    best_specs: dict[str, EngineSpec],
) -> bytes:
    """
    Returns zip bytes containing:
      - leaderboard.csv
      - top5 plots as SVG (and PNG fallback if available)
      - optional HTML interactive plots
    """
    buf = io.BytesIO()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # 1) leaderboard
        csv_bytes = leaderboard_df.to_csv(index=False).encode("utf-8")
        z.writestr(f"leaderboard_{ts}.csv", csv_bytes)

        # 2) plots
        for kind in top5_kinds:
            if kind not in best_bundles or kind not in best_specs:
                continue

            b = best_bundles[kind]
            spec = best_specs[kind]

            sym0 = b.md.symbols()[0]
            bars = b.md.bars[sym0]
            pp = b.report.plots.get("price_panel", {})

            fig = plot_price_indicators_trades_line(
                bars=bars,
                strategy_params=spec.strategy.params,
                indicators=pp.get("indicators"),
                trades=pp.get("trades"),
                indicator_cols=pp.get("indicator_cols"),
                port_cfg=spec.portfolio,
            )

            # Always include interactive HTML (no dependencies, great fallback)
            html = fig.to_html(full_html=True, include_plotlyjs="cdn")
            z.writestr(f"plots/{kind}_price_panel.html", html.encode("utf-8"))
            # 4) closed-trade ledger (PnL per closed trade)
            tables = getattr(b, "report", None).tables if getattr(b, "report", None) is not None else {}
            tl = tables.get("trade_ledger", None)

            if isinstance(tl, pd.DataFrame) and not tl.empty:
                z.writestr(
                    f"ledgers/{kind}_trade_ledger.csv",
                    tl.to_csv(index=False).encode("utf-8"),
                )
            else:
                z.writestr(
                    f"ledgers/{kind}_trade_ledger_EMPTY.txt",
                    b"No trade_ledger table found or it is empty.",
                )


            # Try vector SVG (best for PPT/Word zoom) + high-res PNG fallback
            # Requires kaleido installed.
            try:
                svg_bytes = fig.to_image(format="svg", width=2400, height=1350)
                z.writestr(f"plots/{kind}_price_panel.svg", svg_bytes)
            except Exception as e:
                z.writestr(f"plots/{kind}_SVG_EXPORT_FAILED.txt", str(e).encode("utf-8"))

            try:
                png_bytes = fig.to_image(format="png", width=2400, height=1350, scale=2)
                z.writestr(f"plots/{kind}_price_panel.png", png_bytes)
            except Exception as e:
                z.writestr(f"plots/{kind}_PNG_EXPORT_FAILED.txt", str(e).encode("utf-8"))

    buf.seek(0)
    return buf.getvalue()

def params_string(spec_k: EngineSpec) -> str:
    # Strategy params only (not portfolio)
    p = (spec_k.strategy.params or {})
    # Make stable readable string
    return json.dumps(p, sort_keys=True)
@st.cache_data(show_spinner=False)
def _get_excel_sheets(path: str) -> list[str]:
    try:
        xls = pd.ExcelFile(path)
        return list(xls.sheet_names)
    except Exception:
        return []

def best_params_snapshot(spec: EngineSpec) -> dict:
    # show only params you care about (stable, readable)
    return {
        "strategy": dict(spec.strategy.params or {}),
        "portfolio": {
            "buy_pct_cash": float(getattr(spec.portfolio, "buy_pct_cash", np.nan)),
            "sell_pct_shares": float(getattr(spec.portfolio, "sell_pct_shares", np.nan)),
            "cooldown_bars": int(getattr(spec.portfolio, "cooldown_bars", 0)),
            "min_return_before_sell": float(getattr(spec.portfolio, "min_return_before_sell", 0.0)),
            "sizing_mode": str(getattr(spec.portfolio, "sizing_mode", "")),
            "rebalance_policy": str(getattr(spec.portfolio, "rebalance_policy", "")),
        },
    }

def attach_comparator_to_bundle(
    bundle_a,
    bundle_b,
    *,
    label_a: str,
    label_b: str,
    periods_per_year: int,
    rf_annual: float,
):
    analyzer = ResultsAnalyzer(periods_per_year=periods_per_year, rf_annual=rf_annual)
    comp = analyzer.compare(
        report_a=bundle_a.report,
        report_b=bundle_b.report,
        label_a=label_a,
        label_b=label_b,
    )

    rep0 = bundle_a.report

    new_meta = dict(getattr(rep0, "meta", None) or {})
    new_meta["cum_compare_labels"] = {"a": label_a, "b": label_b}

    new_plots = dict(getattr(rep0, "plots", None) or {})
    new_plots["cum_vs_bench"] = comp["cum_plot"]

    new_tables = dict(getattr(rep0, "tables", None) or {})
    new_tables["curve_vs_comparator"] = comp["curve_table"]

    rep1 = replace(rep0, meta=new_meta, plots=new_plots, tables=new_tables)
    return replace(bundle_a, report=rep1)


import hashlib
import json, hashlib
def _code_version() -> str:
    # include files whose changes should invalidate cached bundles
    files = [
        Path(__file__).parent / "portfolio.py",
        Path(__file__).parent / "results.py",
        Path(__file__).parent / "engine.py",
        Path(__file__).parent / "optimize.py",
        Path(__file__).parent / "indicators.py",
        Path(__file__).parent / "periods.py",
        Path(__file__).parent / "data.py",
        Path(__file__).parent / "strategy.py",

    ]
    parts = []
    for p in files:
        try:
            parts.append(f"{p.name}:{p.stat().st_mtime_ns}")
        except Exception:
            parts.append(f"{p.name}:na")
    return "|".join(parts)

def _spec_key(spec: EngineSpec) -> str:
    d = {
        "data": spec.data.__dict__,
        "strategy": {"kind": spec.strategy.kind, "params": spec.strategy.params},
        "portfolio": spec.portfolio.__dict__,
        "interval": spec.data.interval,
        "code_version": _code_version(), 
    }
    s = json.dumps(d, sort_keys=True, default=str).encode()
    return hashlib.sha1(s).hexdigest()

def leaderboard_row_from_report(report, *, name: str) -> dict:
    """
    Build one leaderboard row from a BacktestReport.
    Uses:
      - report.metrics (Total return, CAGR, Sharpe, Max drawdown, Net PnL)
      - report.tables["trade_performance"] (Trades, Win Rate)
    """
    m = report.metrics or {}
    tp = report.tables.get("trade_performance")

    trades = None
    win_rate = None
    if tp is not None and not tp.empty and "Value" in tp.columns:
        # tp index contains "Trades", "Win Rate", etc.
        try:
            trades = int(float(tp.loc["Trades", "Value"]))
        except Exception:
            trades = None
        try:
            win_rate = float(tp.loc["Win Rate", "Value"])
        except Exception:
            win_rate = None

    # Your metric keys are case-sensitive as defined in ResultsAnalyzer._headline_metrics
    total_return = float(m.get("Total return", float("nan")))
    cagr = float(m.get("CAGR", float("nan")))
    sharpe = float(m.get("Sharpe", float("nan")))
    max_dd = float(m.get("Max drawdown", float("nan")))
    pnl = float(m.get("Net PnL", float("nan")))

    return {
        "Strategy": name,
        "CAGR": cagr,
        "PnL": pnl,
        "Total Return": total_return,
        "# Trades": trades,
        "Win %": (win_rate * 100.0) if (win_rate is not None) else None,
        "Max Drawdown": max_dd,
        "Sharpe": sharpe,
    }


def render_interval_editor_for_kind(kind: str, *, source_key: str, preview_df: pd.DataFrame | None = None):
    """
    UI: lets user select active params + edit domains for ONE strategy kind.
    Persists edits into:
      st.session_state["opt_catalog_by_kind"][kind]
      st.session_state["opt_active_keys_by_kind"][kind]
    """
    catalog = default_param_catalog(kind)
    selectable_keys = list(catalog.keys())

    default_active = (
        ["strategy.sma_fast_window", "strategy.sma_slow_window"] if kind == "ma_cross"
        else ["strategy.sma_window"] if kind == "sma_price"
        else ["strategy.rsi_window", "strategy.rsi_oversold", "strategy.rsi_overbought"] if kind == "rsi"
        else ["strategy.macd_fast_window", "strategy.macd_slow_window", "strategy.macd_signal_window"] if kind == "macd"
        else ["strategy.bb_window", "strategy.bb_k"] if kind == "bollinger"
        else []
    )

    # Load previous edits if they exist
    if "opt_catalog_by_kind" not in st.session_state:
        st.session_state["opt_catalog_by_kind"] = {}
    if "opt_active_keys_by_kind" not in st.session_state:
        st.session_state["opt_active_keys_by_kind"] = {}

    edited_catalog = st.session_state["opt_catalog_by_kind"].get(kind, dict(catalog))
    saved_active = st.session_state["opt_active_keys_by_kind"].get(kind, default_active)

    active_keys = st.multiselect(
        f"[{kind}] Parameters to optimize",
        options=selectable_keys,
        default=[k for k in saved_active if k in selectable_keys],
        key=f"lb_{kind}_active_keys",
    )

    st.markdown(f"**[{kind}] Intervals / choices**")
    # Start from previous edits but ensure all keys exist
    for k in selectable_keys:
        if k not in edited_catalog:
            edited_catalog[k] = catalog[k]

    for k in active_keys:
        pdef = edited_catalog[k]
        st.markdown(f"- **{k}** (`{pdef.kind}`)")

        if pdef.kind == "int":
            lo, hi, step = pdef.domain

            mode = st.radio(
                "Domain mode",
                ["range", "manual list"],
                index=0,
                horizontal=True,
                key=f"lb_{kind}_{k}_mode",
            )

            if mode == "range":
                c1, c2, c3 = st.columns(3)
                lo2 = c1.number_input(f"{k} min", value=int(lo), step=1, key=f"lb_{kind}_{k}_min")
                hi2 = c2.number_input(f"{k} max", value=int(hi), step=1, key=f"lb_{kind}_{k}_max")
                step2 = c3.number_input(f"{k} step", value=int(step), step=1, key=f"lb_{kind}_{k}_step")
                if lo2 > hi2:
                    st.error(f"{kind}.{k}: min must be <= max")
                    st.stop()
                edited_catalog[k] = replace(pdef, domain=(int(lo2), int(hi2), int(step2)))

            else:
                default_txt = default_manual_txt_for_param(k, pdef)
                state_key = f"lb_{kind}_{k}_manual"

                txt = st.text_input(
                    f"{k} values (comma/space separated)",
                    value=st.session_state.get(state_key, default_txt),
                    key=state_key,
                )


                try:
                    vals = parse_int_list(txt)
                except Exception as e:
                    st.error(f"{kind}.{k}: invalid list: {e}")
                    st.stop()

                if not vals:
                    st.error(f"{kind}.{k}: provide at least one value.")
                    st.stop()

                edited_catalog[k] = replace(pdef, kind="choice", domain=vals)

        elif pdef.kind == "float":
            lo, hi, step = pdef.domain
            c1, c2, c3 = st.columns(3)
            lo2 = c1.number_input(f"{k} min", value=float(lo), step=0.01, key=f"lb_{kind}_{k}_min")
            hi2 = c2.number_input(f"{k} max", value=float(hi), step=0.01, key=f"lb_{kind}_{k}_max")
            step2 = c3.number_input(f"{k} step", value=float(step), step=0.01, key=f"lb_{kind}_{k}_step")
            if lo2 > hi2:
                st.error(f"{kind}.{k}: min must be <= max")
                st.stop()
            edited_catalog[k] = replace(pdef, domain=(float(lo2), float(hi2), float(step2)))

        elif pdef.kind == "choice":
            choices = list(pdef.domain)
            picked = st.multiselect(
                f"{k} choices",
                choices,
                default=choices,
                key=f"lb_{kind}_{k}_choices",
            )
            edited_catalog[k] = replace(pdef, domain=list(picked))

        elif pdef.kind == "date_window":
            if source_key != "bmce":
                st.warning("data.window optimization is intended for BMCE uploads.")
                edited_catalog[k] = replace(pdef, domain=[])
            else:
                if preview_df is None:
                    st.warning("Upload/preview BMCE data first to build date windows.")
                    edited_catalog[k] = replace(pdef, domain=[])
                else:
                    min_bars = st.number_input("min bars per window", min_value=30, value=252, step=21, key=f"lb_{kind}_dw_min_bars")
                    step_bars = st.number_input("step bars", min_value=1, value=21, step=1, key=f"lb_{kind}_dw_step_bars")
                    max_windows = st.number_input("max windows", min_value=10, value=200, step=10, key=f"lb_{kind}_dw_max_windows")

                    windows = build_date_windows_from_df(
                        preview_df,
                        min_bars=int(min_bars),
                        step_bars=int(step_bars),
                        max_windows=int(max_windows),
                    )
                    edited_catalog[k] = replace(pdef, domain=windows)

    # persist
    st.session_state["opt_catalog_by_kind"][kind] = edited_catalog
    st.session_state["opt_active_keys_by_kind"][kind] = list(active_keys)


def build_active_params_for_kind(kind: str) -> list[ParamDef]:
    """
    Build active_params for a strategy kind using saved interval edits if available.
    Falls back to default catalog + default active keys otherwise.
    """
    catalog_default = default_param_catalog(kind)

    cat_by_kind = st.session_state.get("opt_catalog_by_kind", {})
    keys_by_kind = st.session_state.get("opt_active_keys_by_kind", {})

    edited_catalog = cat_by_kind.get(kind)
    active_keys = keys_by_kind.get(kind)

    # fallback: use defaults if user never edited this kind
    if edited_catalog is None:
        edited_catalog = dict(catalog_default)

    if not active_keys:
        # fallback default active set (same mapping you used)
        active_keys = (
            ["strategy.sma_fast_window", "strategy.sma_slow_window"] if kind == "ma_cross"
            else ["strategy.sma_window"] if kind == "sma_price"
            else ["strategy.rsi_window", "strategy.rsi_oversold", "strategy.rsi_overbought"] if kind == "rsi"
            else ["strategy.macd_fast_window", "strategy.macd_slow_window", "strategy.macd_signal_window"] if kind == "macd"
            else ["strategy.bb_window", "strategy.bb_k"] if kind == "bollinger"
            else []
        )

    active_params: list[ParamDef] = []
    for k in active_keys:
        if k not in edited_catalog:
            continue
        p = edited_catalog[k]
        if p.enabled and p.domain is not None and (p.kind != "date_window" or len(p.domain) > 0):
            active_params.append(p)

    return active_params


def render_param_catalog_editor(
    catalog: dict[str, ParamDef],
    *,
    prefix: str,
) -> list[ParamDef]:
    """
    Returns a list of ParamDef with updated 'domain' and 'enabled' based on Streamlit inputs.
    """
    edited: list[ParamDef] = []

    for key, p in catalog.items():
        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
        with c1:
            enabled = st.checkbox(key, value=bool(p.enabled), key=f"{prefix}_{key}_en")
        with c2:
            kind = p.kind

        # Only handle int/float here (your catalog uses those)
        if kind == "int":
            lo0, hi0, step0 = p.domain
            with c2:
                lo = st.number_input("min", value=int(lo0), step=1, key=f"{prefix}_{key}_lo")
            with c3:
                hi = st.number_input("max", value=int(hi0), step=1, key=f"{prefix}_{key}_hi")
            with c4:
                step = st.number_input("step", value=int(step0), step=1, key=f"{prefix}_{key}_step")
            dom = (int(lo), int(hi), int(step))

        elif kind == "float":
            lo0, hi0, step0 = p.domain
            with c2:
                lo = st.number_input("min", value=float(lo0), step=float(step0), key=f"{prefix}_{key}_lo")
            with c3:
                hi = st.number_input("max", value=float(hi0), step=float(step0), key=f"{prefix}_{key}_hi")
            with c4:
                step = st.number_input("step", value=float(step0), step=float(step0), key=f"{prefix}_{key}_step")
            dom = (float(lo), float(hi), float(step))

        else:
            # fallback: keep as-is (choice/date_window not used in your default catalog)
            dom = p.domain

        # basic guards
        if enabled:
            if (kind in ("int", "float")) and (dom[2] <= 0 or dom[0] > dom[1]):
                st.warning(f"Invalid domain for {key}. Disabling.")
                enabled = False

        edited.append(replace(p, enabled=enabled, domain=dom))

    return edited


def parse_int_list(s: str) -> list[int]:
    """
    Parse '5,10, 20 50' into [5,10,20,50]. Ignores blanks.
    Raises ValueError on invalid tokens.
    """
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    # allow commas or spaces
    tokens = [t.strip() for t in s.replace(",", " ").split()]
    out = []
    for t in tokens:
        if not t:
            continue
        out.append(int(float(t)))  # lets user type "20.0" too
    # dedupe while preserving order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def add_volume_to_trades_table(
    trades: pd.DataFrame,
    md,  # MarketData
    *,
    volume_col: str = "Volume",
    out_col: str = "volume",
) -> pd.DataFrame:
    """
    Adds bar volume to each fill row using (timestamp, symbol) lookup.
    trades must have columns: ['timestamp','symbol'].
    """
    if trades is None or trades.empty:
        return trades
    if "timestamp" not in trades.columns or "symbol" not in trades.columns:
        return trades

    tdf = trades.copy()
    tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], errors="coerce")
    tdf = tdf.dropna(subset=["timestamp", "symbol"])

    # Build a lookup table: (timestamp, symbol) -> volume
    parts = []
    for sym, bars in md.bars.items():
        if volume_col not in bars.columns:
            continue
        b = bars[[volume_col]].copy()
        b = b.sort_index()
        b = b.reset_index().rename(columns={b.index.name or "index": "timestamp", volume_col: out_col})
        b["timestamp"] = pd.to_datetime(b["timestamp"], errors="coerce")
        b["symbol"] = sym
        parts.append(b[["timestamp", "symbol", out_col]])

    if not parts:
        return tdf

    vol_df = pd.concat(parts, ignore_index=True)
    # Exact join on timestamp+symbol (your fills timestamp should match bars index)
    tdf = tdf.merge(vol_df, on=["timestamp", "symbol"], how="left")

    return tdf

def _persist_upload_to_cache(uploaded_file, tag: str, symbol: str) -> tuple[str, str]:
    """
    Persist an UploadedFile to a stable on-disk path based on content hash.
    Returns (path, sha1_hex).
    """
    data = uploaded_file.getvalue()
    h = hashlib.sha1(data).hexdigest()
    suffix = Path(uploaded_file.name).suffix.lower() or ".dat"
    safe_sym = "".join(ch for ch in (symbol or "SYM") if ch.isalnum() or ch in ("_", "-"))[:32]
    cache_root = Path.home() / ".backtester_cache" / "uploads"
    cache_root.mkdir(parents=True, exist_ok=True)
    out_path = cache_root / f"{tag}_{safe_sym}_{h}{suffix}"
    if (not out_path.exists()) or (out_path.stat().st_size != len(data)):
        out_path.write_bytes(data)
    return str(out_path), h





# ============================================================
# Plotting helpers (NORMAL price line + indicators + buy/sell)
# ============================================================
@st.cache_data(show_spinner=False)
def load_benchmark_market_data_cached(
    bench_source_key: str,
    bench_symbol: str,
    timezone: str,
    interval: str,
    bmce_path: str | None,
    start: str | None,
    end: str | None,
    yf_period: str | None,
    yf_interval: str | None,
    yf_auto_adjust: bool | None,
):
    if bench_source_key == "bmce":
        if bmce_path is None:
            raise ValueError("Benchmark BMCE selected but no file path provided.")
        ds = BMCEDataSource(timezone=timezone)
        md = ds.load(
            symbols=[bench_symbol],
            start=start,
            end=end,
            interval=interval,
            paths=bmce_path,
        )
        return md

    # yfinance
    ds = YahooFinanceDataSource(timezone=timezone)
    md = ds.load(
        symbols=[bench_symbol],
        start=start,
        end=end,
        interval=(yf_interval or "1d"),
        auto_adjust=bool(yf_auto_adjust),
        progress=False,
    )
    return md



from plotly.subplots import make_subplots

def plot_price_indicators_trades_line(
    bars: pd.DataFrame,
    strategy_params: dict | None = None,
    indicators: pd.DataFrame | None = None,
    trades: pd.DataFrame | None = None,
    indicator_cols: list[str] | None = None,
    *,
    port_cfg: PortfolioConfig | None = None,
    rsi_low: float = 30.0,
    rsi_high: float = 70.0,
) -> go.Figure:
    """
    Price (row 1), RSI and/or MACD panels (middle rows), Volume (last row).

    - RSI is plotted in its own panel with threshold lines + shaded regions.
    - MACD panel plots EMA_fast, EMA_slow + histogram (EMA_fast-EMA_slow) as bars
      with per-bar green/red intensity (stronger color as magnitude increases).
    """

    df = bars.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    ind = None
    if indicators is not None and not indicators.empty:
        ind = indicators.copy()
        if not isinstance(ind.index, pd.DatetimeIndex):
            ind.index = pd.to_datetime(ind.index)
        ind = ind.reindex(df.index)

        if indicator_cols is not None:
            keep = [c for c in indicator_cols if c in ind.columns]
            if keep:
                ind = ind[keep]

    def _maybe_add_bollinger_overlay():
        nonlocal fig, df, ind, indicator_cols

        if ind is None or ind.empty:
            return

        cols = list(ind.columns)

        # --- HARD GATE: only plot BB if explicitly requested ---
        is_bollinger_strategy = bool(strategy_params) and (
            ("bb_k" in strategy_params) or ("bb_window" in strategy_params)
        )

        has_std_col = any(c.startswith("std_") for c in cols)
        has_std_requested = bool(indicator_cols) and any(c.startswith("std_") for c in indicator_cols)

        # If it's not a bollinger strategy and no std column is present/requested, do nothing
        if not (is_bollinger_strategy or has_std_col or has_std_requested):
            return

        # From here on, infer w ONLY from std_ first (preferred), then sma_ if needed
        w = None

        # Prefer explicit requested std_
        if indicator_cols:
            for c in indicator_cols:
                if c.startswith("std_"):
                    try:
                        w = int(c.split("_", 1)[1])
                        break
                    except Exception:
                        pass

        # Else infer from actual std_ columns present
        if w is None:
            for c in cols:
                if c.startswith("std_"):
                    try:
                        w = int(c.split("_", 1)[1])
                        break
                    except Exception:
                        pass

        # If still none, fall back to SMA window (only if bollinger strategy)
        if w is None and is_bollinger_strategy:
            for c in cols:
                if c.startswith("sma_"):
                    try:
                        w = int(c.split("_", 1)[1])
                        break
                    except Exception:
                        pass

        if w is None:
            return

        k = float((strategy_params or {}).get("bb_k", 2.0))
        col_mid = f"sma_{w}"
        col_std = f"std_{w}"

        if col_mid not in ind.columns:
            return

        mid = pd.to_numeric(ind[col_mid], errors="coerce")

        if col_std in ind.columns:
            std = pd.to_numeric(ind[col_std], errors="coerce")
        else:
            # Only allow computing std fallback if this is explicitly bollinger strategy
            if not is_bollinger_strategy:
                return
            std = pd.to_numeric(df["Close"], errors="coerce").rolling(int(w), min_periods=int(w)).std()

        upper = mid + k * std
        lower = mid - k * std

        # Add mid/upper/lower on PRICE row (row=1)
        fig.add_trace(
            go.Scatter(x=df.index, y=mid, mode="lines", name=f"BB mid (SMA{w})", line=dict(width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=upper, mode="lines", name=f"BB upper (k={k})", line=dict(width=1, dash="dash")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=lower, mode="lines", name=f"BB lower (k={k})", line=dict(width=1, dash="dash")),
            row=1, col=1
        )

        # Optional fill between upper and lower (nice visual)
        fig.add_trace(
            go.Scatter(
                x=df.index, y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=lower,
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                name="BB band",
                opacity=0.12,
                hoverinfo="skip",
            ),
            row=1, col=1
        )

    # ----------------------------
    # Detect RSI / MACD columns
    # ----------------------------
    rsi_cols: list[str] = []
    macd_bases: list[str] = []
    has_rsi = False
    has_macd = False


    if ind is not None and not ind.empty:
        cols = list(ind.columns)

        # RSI columns: rsi_{n}
        rsi_cols = [c for c in cols if c.startswith("rsi_")]

        # MACD base: macd_{fast}_{slow}_{sig} (but in DF you likely have __line/__signal/__hist)
        # We'll infer unique "macd_{fast}_{slow}_{sig}" bases from columns.
        for c in cols:
            if c.startswith("macd_") and "__" in c:
                base = c.split("__", 1)[0]
                if base not in macd_bases:
                    macd_bases.append(base)

        # Pick RSI column that matches strategy window (or indicator_cols), else fallback
        rsi_col = None
        if ind is not None and (not ind.empty):
            rsi_cols = [c for c in ind.columns if c.startswith("rsi_")]

            if rsi_cols:
                # 1) strategy-driven choice
                w = (strategy_params or {}).get("rsi_window", None)
                if w is not None:
                    cand = f"rsi_{int(w)}"
                    if cand in ind.columns:
                        rsi_col = cand

                # 2) indicator_cols-driven choice (if provided)
                if rsi_col is None and indicator_cols:
                    for c in indicator_cols:
                        if c in ind.columns and c.startswith("rsi_"):
                            rsi_col = c
                            break

                # 3) fallback: first available
                if rsi_col is None:
                    rsi_col = rsi_cols[0]

            # MACD bases
            macd_bases = []
            for c in ind.columns:
                if c.startswith("macd_") and "__" in c:
                    base = c.split("__", 1)[0]
                    if base not in macd_bases:
                        macd_bases.append(base)

        # flags (ALWAYS defined because initialized above)
        has_rsi = (rsi_col is not None)
        has_macd = (len(macd_bases) > 0)

        macd_base = macd_bases[0] if has_macd else None



    # ----------------------------
    # Layout: rows depend on panels
    # ----------------------------
    # row 1: price
    # optional row 2: RSI
    # optional row 3: MACD
    # last row: volume
    n_mid = int(has_rsi) + int(has_macd)
    n_rows = 2 + n_mid  # price + volume + middle panels

    # heights: price biggest, then mid panels, then volume
    # normalize later by relative weights
    row_heights = []
    row_heights.append(0.58)  # price
    if has_rsi:
        row_heights.append(0.20)
    if has_macd:
        row_heights.append(0.20)
    row_heights.append(0.22)  # volume

    # normalize to sum=1
    s = sum(row_heights)
    row_heights = [h / s for h in row_heights]

    # Determine which row is MACD row (if present)
    macd_row = None
    tmp_row = 2
    if has_rsi:
        tmp_row += 1
    if has_macd:
        macd_row = tmp_row  # the current row where MACD panel is plotted

    specs = [[{}] for _ in range(n_rows)]
    if macd_row is not None:
        specs[macd_row - 1][0] = {"secondary_y": True}  # plotly is 1-indexed rows

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        specs=specs,
    )


    # =========================
    # Row 1: Price + Trades
    # =========================
    fig.add_trace(
        go.Scatter(x=df.index, y=df["Close"].astype(float), mode="lines", name="Close"),
        row=1, col=1
    )
    _maybe_add_bollinger_overlay()
    if ind is not None and not ind.empty:
        # Keep it conservative: only draw moving averages on the price panel
        price_level_cols = [c for c in ind.columns if c.startswith(("sma_", "ema_"))]

        for c in price_level_cols:
            y = pd.to_numeric(ind[c], errors="coerce")
            if y.notna().any():  # avoid adding empty traces
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=y,
                        mode="lines",
                        name=c,
                        line=dict(width=1),
                    ),
                    row=1, col=1
                )

    # Trades markers on row 1
    if trades is not None and not trades.empty:
        t = trades.copy()
        t["timestamp"] = pd.to_datetime(t["timestamp"], errors="coerce")
        t = t.dropna(subset=["timestamp"]).sort_values("timestamp")

        if "side" not in t.columns:
            t["side"] = np.where(pd.to_numeric(t["qty"], errors="coerce").fillna(0) > 0, "BUY", "SELL")

        if "price" in t.columns:
            y = pd.to_numeric(t["price"], errors="coerce")
        else:
            y = pd.Series(np.nan, index=t.index)

        close_map = df["Close"].reindex(t["timestamp"]).ffill()
        t["y_plot"] = np.where(np.isfinite(y.values), y.values, close_map.values)

        buys = t[t["side"] == "BUY"]
        sells = t[t["side"] == "SELL"]

        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["timestamp"],
                    y=buys["y_plot"],
                    mode="markers+text",
                    marker=dict(
                        size=14,
                        symbol="triangle-up",
                        color="#00B050",          # vert clair
                        line=dict(color="#004D1A", width=1.5),  # contour sombre
                    ),
                    text=["BUY"] * len(buys),
                    textposition="top center",
                    textfont=dict(size=12, color="#004D1A"),
                    name="BUY",
                ),
                row=1, col=1
            )

        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["timestamp"],
                    y=sells["y_plot"],
                    mode="markers+text",
                    marker=dict(
                        size=14,
                        symbol="triangle-down",
                        color="#C00000",          # rouge clair
                        line=dict(color="#4D0000", width=1.5),
                    ),
                    text=["SELL"] * len(sells),
                    textposition="bottom center",
                    textfont=dict(size=12, color="#4D0000"),
                    name="SELL",
                ),
                row=1, col=1
            )


    # =========================
    # Middle panels: RSI / MACD
    # =========================
    cur_row = 2  # first middle row

    # ---- RSI panel ----
    if has_rsi and rsi_col is not None:
        rsi = pd.to_numeric(ind[rsi_col], errors="coerce")

        # main RSI line
        fig.add_trace(
            go.Scatter(x=df.index, y=rsi, mode="lines", name=rsi_col),
            row=cur_row, col=1
        )

        # thresholds: prefer strategy_params (optimized) else fall back to function defaults
        lo = float((strategy_params or {}).get("rsi_oversold", rsi_low))
        hi = float((strategy_params or {}).get("rsi_overbought", rsi_high))

        # guard: ensure lo < hi (avoid broken shading/lines if user misconfigures)
        if lo >= hi:
            lo, hi = min(lo, hi - 1e-9), hi  # or just set lo=30,hi=70; your choice

        # threshold lines
        fig.add_hline(y=lo, line_width=1, line_dash="dash", row=cur_row, col=1)
        fig.add_hline(y=hi, line_width=1, line_dash="dash", row=cur_row, col=1)

        # shaded regions
        fig.add_hrect(y0=0, y1=lo, row=cur_row, col=1, opacity=0.12, line_width=0)
        fig.add_hrect(y0=hi, y1=100, row=cur_row, col=1, opacity=0.12, line_width=0)


        # keep RSI scale consistent
        fig.update_yaxes(range=[0, 100], title_text="RSI", row=cur_row, col=1)

        cur_row += 1

    # ---- MACD panel ----
    if has_macd and macd_base is not None:
        line_col = f"{macd_base}__line"
        sig_col  = f"{macd_base}__signal"
        hist_col = f"{macd_base}__hist"

        line = pd.to_numeric(ind.get(line_col), errors="coerce")
        sigl = pd.to_numeric(ind.get(sig_col), errors="coerce")

        # histogram: prefer __hist if you have it, else line - signal
        if hist_col in ind.columns:
            hist = pd.to_numeric(ind[hist_col], errors="coerce")
        else:
            hist = line - sigl

        # MACD line + signal line (PRIMARY y)
        fig.add_trace(
            go.Scatter(x=df.index, y=line, mode="lines", name=f"{macd_base} line"),
            row=cur_row, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=sigl, mode="lines", name=f"{macd_base} signal"),
            row=cur_row, col=1, secondary_y=False
        )

        # Histogram bars (SECONDARY y)
        h = hist.fillna(0.0).astype(float)
        bar_colors = np.where(h >= 0, "rgba(0,180,0,0.6)", "rgba(200,0,0,0.6)")

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=h.values,
                name=f"{macd_base} hist",
                marker=dict(color=bar_colors),
            ),
            row=cur_row, col=1, secondary_y=True
        )

        # Zero line on histogram axis (SECONDARY y)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=np.zeros(len(df.index)),
                mode="lines",
                showlegend=False,
            ),
            row=cur_row, col=1, secondary_y=True
        )

        fig.update_yaxes(title_text="MACD", row=cur_row, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Hist", row=cur_row, col=1, secondary_y=True)

        cur_row += 1


    # =========================
    # Last Row: Volume + Gate/Cap
    # =========================
    vol_row = n_rows

    if port_cfg is not None:
        vcol = getattr(port_cfg, "volume_col", "Volume")
    else:
        vcol = "Volume"

    if vcol in df.columns:
        vol = pd.to_numeric(df[vcol], errors="coerce").fillna(0.0).astype(float)

        fig.add_trace(
            go.Bar(x=df.index, y=vol.values, name="Volume"),
            row=vol_row, col=1
        )

        def _adv(series: pd.Series, w: int) -> pd.Series:
            w = int(max(1, w))
            return series.rolling(w, min_periods=1).mean()

        if port_cfg is not None:
            if bool(getattr(port_cfg, "use_volume_gate", False)):
                kind = str(getattr(port_cfg, "volume_gate_kind", "min_abs"))
                if kind == "min_abs":
                    gate_val = float(getattr(port_cfg, "min_volume_abs", 0.0))
                    gate_line = pd.Series(gate_val, index=df.index)
                else:
                    ratio = float(getattr(port_cfg, "min_volume_ratio_adv", 0.0))
                    w = int(getattr(port_cfg, "volume_gate_adv_window", 20))
                    gate_line = ratio * _adv(vol, w)

                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=gate_line.values,
                        mode="lines",
                        name="Gate",
                        line=dict(width=3),
                        showlegend=True,
                    ),
                    row=vol_row, col=1
                )

            if bool(getattr(port_cfg, "use_participation_cap", False)):
                pr = float(getattr(port_cfg, "participation_rate", 0.05))
                basis = str(getattr(port_cfg, "participation_basis", "bar"))
                if basis == "bar":
                    liq = vol
                else:
                    w = int(getattr(port_cfg, "adv_window", 20))
                    liq = _adv(vol, w)

                cap_line = pr * liq

                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=cap_line.values,
                        mode="lines",
                        name="Cap",
                        line=dict(width=3, dash="dash"),
                        showlegend=True,
                    ),
                    row=vol_row, col=1
                )

    # ----------------------------
    # Layout polish
    # ----------------------------
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        font=dict(family="Arial", size=12),
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=60, b=40),
        bargap=0.0,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=vol_row, col=1)

    return fig



def plot_cum_vs_bench(
    cum: pd.Series,
    bench_cum: pd.Series | None,
    *,
    label_a: str = "Strategy",
    label_b: str = "Comparator",
    title: str = "Cumulative Returns",
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 4))
    ax = plt.gca()

    ax.plot(cum.index, cum.values, label=label_a)
    if bench_cum is not None:
        ax.plot(bench_cum.index, bench_cum.values, label=label_b)

    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    return fig


def plot_drawdown_red(dd: pd.Series) -> plt.Figure:
    fig = plt.figure(figsize=(14, 3))
    ax = plt.gca()
    ax.plot(dd.index, dd.values, label="Drawdown")
    ax.fill_between(dd.index, dd.values, 0.0, where=(dd.values < 0.0), alpha=0.35)
    ax.set_title("Drawdown")
    ax.grid(True)
    ax.legend()
    return fig


def plot_monthly_heatmap_with_values(monthly: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(14, 5))
    ax = plt.gca()

    if monthly is None or monthly.empty:
        ax.set_title("Monthly returns heatmap (no data)")
        return fig

    data = monthly.values.astype(float)
    if np.isnan(data).all():
        ax.set_title("Monthly returns heatmap (no data)")
        return fig

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    if vmin < 0 < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(data, aspect="auto", norm=norm, cmap="RdYlGn")
    ax.set_title("Monthly Returns Heatmap")
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index.astype(str).tolist())
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    return fig


def plot_yearly_returns_bar(yearly: pd.Series) -> plt.Figure:
    fig = plt.figure(figsize=(14, 3))
    ax = plt.gca()
    if yearly is None or yearly.empty:
        ax.set_title("Yearly returns (no data)")
        return fig
    years = yearly.index.astype(str).tolist()
    vals = yearly.values.astype(float)
    ax.bar(years, vals)
    ax.set_title("Yearly Returns")
    ax.grid(True, axis="y")
    return fig





def render_comparison(
    bundle_a,
    bundle_b,
    *,
    label_a: str = "Strategy",
    label_b: str = "Benchmark",
    periods_per_year: int = 252,
    rf_annual: float = 0.0,
):
    analyzer = ResultsAnalyzer(periods_per_year=periods_per_year, rf_annual=rf_annual)

    # You will implement analyzer.compare(...) in results.py
    comp = analyzer.compare(
        report_a=bundle_a.report,
        report_b=bundle_b.report,
        label_a=label_a,
        label_b=label_b,
    )

    st.subheader("Cumulative Returns vs Comparator")
    cvb = comp["cum_plot"]
    st.pyplot(plot_cum_vs_bench(cvb["strategy"], cvb["benchmark"]))

    st.subheader("Curve vs Comparator")
    st.dataframe(comp["curve_table"], use_container_width=True)


# ============================================================
# Render bundle
# ============================================================

def render_bundle(bundle, *, port_cfg: PortfolioConfig | None = None, label : str| None = None,strategy_params: dict | None = None,   # ✅ NEW
):
    rep = bundle.report
    plots = rep.plots
    tables = rep.tables
    st.subheader("Price + Indicators + Trades")
    pp = plots["price_panel"]
    sym0 = bundle.md.symbols()[0]
    bars = bundle.md.bars[sym0]

    fig = plot_price_indicators_trades_line(
        bars=bars,
        strategy_params=strategy_params,
        indicators=pp.get("indicators"),
        trades=pp.get("trades"),
        indicator_cols=pp.get("indicator_cols"),
        port_cfg=port_cfg,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cumulative Returns vs Comparator")
    cvb = plots["cum_vs_bench"]

    sym0 = bundle.md.symbols()[0] if hasattr(bundle, "md") and bundle.md.symbols() else ""
    kind = getattr(getattr(bundle, "spec", None), "strategy", None)
    kind = getattr(kind, "kind", "strategy")

    labels = rep.meta.get("cum_compare_labels", {}) if hasattr(rep, "meta") else {}
    label_a = labels.get("a", f"{kind} | {sym0}")
    label_b = labels.get("b", "Comparator")

    st.pyplot(
        plot_cum_vs_bench(
            cvb["strategy"],
            cvb.get("benchmark"),
            label_a=label_a,
            label_b=label_b,
        )
    )


    st.subheader("Drawdown")
    st.pyplot(plot_drawdown_red(plots["drawdown"]))


    st.subheader("Monthly Returns Heatmap")
    st.pyplot(plot_monthly_heatmap_with_values(plots["monthly_heatmap"]))

    st.subheader("Yearly Returns")
    st.pyplot(plot_yearly_returns_bar(plots["yearly_bar"]))

    st.subheader("Trades (fills)")
    if "trades" in tables:
        trades_df = tables["trades"]

        # pick the right volume column name
        vcol = getattr(port_cfg, "volume_col", "Volume") if port_cfg is not None else "Volume"

        trades_df = add_volume_to_trades_table(trades_df, bundle.md, volume_col=vcol, out_col="volume")

        # optional: choose visible columns order
        cols_first = [c for c in ["timestamp","symbol","side","qty","price","notional","cost","volume"] if c in trades_df.columns]
        cols_rest = [c for c in trades_df.columns if c not in cols_first]
        trades_df = trades_df[cols_first + cols_rest]

        st.dataframe(trades_df, use_container_width=True)


    st.subheader("Trade Ledger (PnL per closed trade)")
    if "trade_ledger" in tables:
        st.dataframe(tables["trade_ledger"], use_container_width=True)

    st.subheader("Trade Performance (summary)")
    if "trade_performance" in tables:
        st.dataframe(tables["trade_performance"], use_container_width=True)
    if "curve_vs_comparator" in tables:
        st.subheader("Curve vs Comparator")
        st.dataframe(tables["curve_vs_comparator"], use_container_width=True)




# ============================================================
# Spec builder
# ============================================================

def make_base_spec(
    source_key: str,
    symbol: str,
    timezone: str,
    interval: str,
    bmce_tmp_path: Optional[str],
    start: Optional[str],
    end: Optional[str],
    include_windows: Optional[list[tuple[str, str]]],
    exclude_windows: Optional[list[tuple[str, str]]],
    yf_period: Optional[str],
    yf_interval: Optional[str],
    yf_auto_adjust: Optional[bool],
    strategy_kind: str,
    strategy_params: dict,
    allow_short: bool,
    initial_cash: float,
    rebalance_policy: str,
    sizing_mode: str,
    buy_pct_cash: float,
    sell_pct_shares: float,
    cooldown_bars: int,
    cost_model: CostModel,
    use_volume_gate: bool,
    volume_gate_kind: str,
    min_volume_abs: float,
    min_volume_ratio_adv: float,
    volume_gate_adv_window: int,
    use_participation_cap: bool,
    participation_rate: float,
    participation_basis: str,
    adv_window: int,
    min_return_before_sell: float,
) -> EngineSpec:

    if source_key == "bmce":
        if not bmce_tmp_path:
            raise ValueError("BMCE selected but bmce_tmp_path is None/empty.")
        data_cfg = DataConfig(
            source="bmce",
            symbols=[symbol],
            timezone=timezone,
            interval=interval,
            start=start,
            end=end,
            include_windows=include_windows,
            exclude_windows=exclude_windows,
            bmce_paths=bmce_tmp_path,
        )

    else:
        data_cfg = DataConfig(
            source="yfinance",
            symbols=[symbol],
            timezone=timezone,
            interval=interval,
            start=start,
            end=end,
            include_windows=include_windows,
            exclude_windows=exclude_windows,
            yf_period=yf_period or "max",
            yf_interval=yf_interval or "1d",
            yf_auto_adjust=bool(yf_auto_adjust),
        )

    strat_cfg = StrategyConfig(kind=strategy_kind, params=dict(strategy_params or {}))
    ind_cfg = IndicatorsConfig(specs=None)

    port_cfg = PortfolioConfig(
        allow_short=bool(allow_short),
        initial_cash=float(initial_cash),
        rebalance_policy=str(rebalance_policy),
        cost_model=cost_model,
        sizing_mode=str(sizing_mode),
        buy_pct_cash=float(buy_pct_cash),
        sell_pct_shares=float(sell_pct_shares),
        min_return_before_sell=float(min_return_before_sell),
        cooldown_bars=int(cooldown_bars),
        use_volume_gate=bool(use_volume_gate),
        volume_gate_kind=str(volume_gate_kind),
        min_volume_abs=float(min_volume_abs),
        min_volume_ratio_adv=float(min_volume_ratio_adv),
        volume_gate_adv_window=int(volume_gate_adv_window),

        use_participation_cap=bool(use_participation_cap),
        participation_rate=float(participation_rate),
        participation_basis=str(participation_basis),
        adv_window=int(adv_window),

    )

    return EngineSpec(
        data=data_cfg,
        indicators=ind_cfg,
        strategy=strat_cfg,
        portfolio=port_cfg,
        periods_per_year=252,
        rf_annual=0.0,
    )


# ============================================================
# BMCE window generator (for optimizing data.window)
# ============================================================

def build_date_windows_from_df(
    df: pd.DataFrame,
    date_col_candidates=("Date", "date", "timestamp", "Datetime", "DATE"),
    min_bars: int = 252,
    step_bars: int = 21,
    max_windows: int = 200,
) -> List[Tuple[str, str]]:
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    dts = pd.to_datetime(df[date_col], errors="coerce")
    dts = dts.dropna().sort_values().reset_index(drop=True)
    if len(dts) < min_bars:
        return []

    out = []
    start = 0
    while True:
        end = start + min_bars - 1
        if end >= len(dts):
            break
        s = dts.iloc[start].date().isoformat()
        e = dts.iloc[end].date().isoformat()
        out.append((s, e))
        if len(out) >= max_windows:
            break
        start += step_bars
    return out


# ============================================================
# Session state helpers
# ============================================================

def ss_get(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


# ============================================================
# Sidebar: Data source
# ============================================================

st.sidebar.header("Data source")
source = st.sidebar.selectbox("Source", ["(upload)", "yfinance","(local file)"], index=0)
source_key = "bmce" if source.startswith("(") else "yfinance"

# Keep a stable symbol in session_state
if "main_symbol" not in st.session_state:
    st.session_state["main_symbol"] = "IAM" if source_key == "bmce" else "AAPL"
symbol = st.session_state["main_symbol"]
timezone = st.sidebar.text_input("Timezone", value="GMT")
interval = st.sidebar.selectbox("Interval", ["1d"], index=0)

bmce_file = None
yf_period = yf_interval = None
yf_auto_adjust = None

if source_key == "bmce":
    bmce_file = None
    bmce_path = None  # <- this is what we will pass into DataConfig.bmce_paths

    if source == "(local file)":
        # Put your default IAM file somewhere in your project, e.g. ./data/IAM.xlsx
        default_path = str((Path(__file__).parent / "data" / "Data IAM.xlsx").resolve())
        bmce_path = st.sidebar.text_input("Local BMCE file path (CSV/XLSX)", value=default_path)

    else:  # bmce (upload)
        bmce_file = st.sidebar.file_uploader("Upload BMCE CSV/XLSX", type=["csv", "xlsx"])
        if bmce_file is not None:
            # Persist upload WITHOUT relying on the current symbol (important for multi-sheet workbooks)
            bmce_cached_path, bmce_file_hash = _persist_upload_to_cache(
                bmce_file, tag="bmce", symbol="WORKBOOK"
            )
            st.session_state["bmce_cached_path"] = bmce_cached_path
            st.session_state["bmce_file_hash"] = bmce_file_hash
            bmce_path = bmce_cached_path

        # After upload (or if already cached), choose ticker:
        if bmce_path and bmce_path.lower().endswith((".xlsx", ".xls")):
            sheets = _get_excel_sheets(bmce_path)
            if sheets:
                symbol = st.sidebar.selectbox(
                    "Ticker (sheet)",
                    options=sheets,
                    index=(sheets.index(st.session_state.get("main_symbol")) if st.session_state.get("main_symbol") in sheets else 0),
                    key="main_sheet_select",
                )
            else:
                symbol = st.sidebar.text_input("Symbol", value=st.session_state.get("main_symbol", "IAM"), key="main_symbol_text")
        else:
            symbol = st.sidebar.text_input("Symbol", value=st.session_state.get("main_symbol", "IAM"), key="main_symbol_text")

        # Store back
        st.session_state["main_symbol"] = symbol


else:
    st.sidebar.caption("yfinance needs internet + yfinance in requirements.txt")
    yf_period = st.sidebar.text_input("yfinance period", value="5y")
    yf_interval = st.sidebar.selectbox("yfinance interval", ["1d"], index=0)
    yf_auto_adjust = st.sidebar.checkbox("auto_adjust", value=False)

st.sidebar.header("Strategy")
strategy_kind = st.sidebar.selectbox("Strategy kind", ["ma_cross", "sma_price", "rsi", "macd", "bollinger"], index=0)
allow_short = st.sidebar.checkbox("Allow short", value=False)

# Reset optimization UI on strategy change (prevents stale widget keys)
prev_kind = ss_get("prev_strategy_kind", strategy_kind)
if prev_kind != strategy_kind:
    for k in list(st.session_state.keys()):
        if k.endswith("_min") or k.endswith("_max") or k.endswith("_step") or k.endswith("_choices"):
            del st.session_state[k]
    st.session_state["prev_strategy_kind"] = strategy_kind

st.sidebar.header("Comparison")

compare_mode = st.sidebar.selectbox(
    "Compare against",
    ["None", "Buy & Hold (same data)", "Another strategy (same data)"],
    index=0,
)

compare_strategy_kind = None
compare_params = {}

if compare_mode == "Another strategy (same data)":
    compare_strategy_kind = st.sidebar.selectbox(
        "Comparator strategy kind",
        ["buy_hold", "ma_cross", "sma_price", "rsi", "macd", "bollinger"],
        index=0,
        key="cmp_kind",
    )
    with st.expander("Comparator parameters", expanded=True):
        compare_params = render_strategy_params_ui(compare_strategy_kind, prefix="cmp", allow_short=allow_short)



# ============================================================
# Sidebar: Benchmark (optional)
# ============================================================
st.sidebar.header("Benchmark (optional)")
use_benchmark = st.sidebar.checkbox("Enable benchmark", value=False)

bench_source_key = None
bench_symbol = None
bench_bmce_file = None
bench_yf_period = None
bench_yf_interval = None
bench_yf_auto_adjust = None

if use_benchmark:
    bench_mode = st.sidebar.selectbox(
        "Benchmark source",
        ["same as main source", "yfinance", "bmce (upload)"],
        index=0
    )

    if bench_mode == "same as main source":
        bench_source_key = source_key
        bench_symbol = st.sidebar.text_input("Benchmark symbol", value="MASI" if source_key == "bmce" else "SPY")

        if bench_source_key == "bmce":
            bench_bmce_file = st.sidebar.file_uploader("Upload BMCE benchmark CSV/XLSX", type=["csv", "xlsx"], key="bench_bmce")
            if bench_bmce_file is not None:
                bench_cached_path, bench_file_hash = _persist_upload_to_cache(
                    bench_bmce_file, tag="bmce_bench", symbol="WORKBOOK"
                )
                st.session_state["bench_cached_path"] = bench_cached_path
                st.session_state["bench_file_hash"] = bench_file_hash

            bench_path = st.session_state.get("bench_cached_path")

            if bench_path and bench_path.lower().endswith((".xlsx", ".xls")):
                sheets = _get_excel_sheets(bench_path)
                if sheets:
                    bench_symbol = st.sidebar.selectbox("Benchmark ticker (sheet)", sheets, key="bench_sheet_select")
            else:
                # keep your existing bench_symbol text_input behavior
                pass

        else:
            bench_yf_period = st.sidebar.text_input("Benchmark yfinance period", value=yf_period or "5y", key="bench_yf_period")
            bench_yf_interval = st.sidebar.selectbox("Benchmark yfinance interval", ["1d"], index=0, key="bench_yf_interval")
            bench_yf_auto_adjust = st.sidebar.checkbox("Benchmark auto_adjust", value=bool(yf_auto_adjust), key="bench_yf_auto_adjust")

    elif bench_mode == "yfinance":
        bench_source_key = "yfinance"
        bench_symbol = st.sidebar.text_input("Benchmark symbol", value="SPY")
        bench_yf_period = st.sidebar.text_input("Benchmark yfinance period", value="5y")
        bench_yf_interval = st.sidebar.selectbox("Benchmark yfinance interval", ["1d"], index=0)
        bench_yf_auto_adjust = st.sidebar.checkbox("Benchmark auto_adjust", value=False)

    else:  # bmce upload
        bench_source_key = "bmce"
        bench_symbol = st.sidebar.text_input("Benchmark symbol", value="MASI")
        bench_bmce_file = st.sidebar.file_uploader("Upload BMCE benchmark CSV/XLSX", type=["csv", "xlsx"], key="bench_bmce2")

        if bench_bmce_file is not None:
            bench_cached_path, bench_file_hash = _persist_upload_to_cache(bench_bmce_file, tag="bmce_bench", symbol=bench_symbol)
            st.session_state["bench_cached_path"] = bench_cached_path
            st.session_state["bench_file_hash"] = bench_file_hash

st.sidebar.header("Periods (include/exclude)")

use_period_filters = st.sidebar.checkbox("Enable period filters", value=False)

include_windows: list[tuple[str, str]] | None = None
exclude_windows: list[tuple[str, str]] | None = None

if use_period_filters:
    st.sidebar.subheader("Market periods (MASI-level)")

    market_include = st.sidebar.multiselect(
        "Include market periods",
        options=list(MASI_PERIODS.keys()),
        default=list(MASI_PERIODS.keys()),
    )
    market_exclude = st.sidebar.multiselect(
        "Exclude market periods",
        options=list(MASI_PERIODS.keys()),
        default=[],
        key="market_exclude",
    )

    # Build include windows from selection (if user unselects all, treat as "no include restriction")
    inc = [MASI_PERIODS[k] for k in market_include] if market_include else []
    exc = [MASI_PERIODS[k] for k in market_exclude] if market_exclude else []

    st.sidebar.subheader("Stock-specific overlays")

    if symbol.upper() == "IAM":
        iam_include = st.sidebar.multiselect(
            "Include IAM periods (optional)",
            options=list(IAM_PERIODS.keys()),
            default=[],
            key="iam_include",
        )
        iam_exclude = st.sidebar.multiselect(
            "Exclude IAM periods",
            options=list(IAM_PERIODS.keys()),
            default=[],
            key="iam_exclude",
        )

        inc_i = [IAM_PERIODS[k] for k in iam_include] if iam_include else []
        exc_i = [IAM_PERIODS[k] for k in iam_exclude] if iam_exclude else []

        # Methodology:
        # 1) include = market includes (if any were selected)
        # 2) if IAM include selected, we further restrict by intersecting -> easiest way is to add to include_windows
        #    and let engine do union-of-includes; to get true intersection you either:
        #      - do it in engine (more complex), or
        #      - choose to interpret "IAM include" as an additional allowed window set.
        # Here we implement "IAM include" as additional includes; if you want strict intersection later, we can upgrade.
        inc = inc + inc_i
        exc = exc + exc_i

    include_windows = inc if inc else None
    exclude_windows = exc if exc else None


# ============================================================
# Main preview (BMCE)
# ============================================================

preview_df = None
if source_key == "bmce":
    st.subheader("BMCE data")

    if source == "(upload)" and bmce_file is None:
        st.info("Upload a BMCE CSV/XLSX.")
        st.stop()

    if bmce_path is None:
        st.error("No BMCE path provided.")
        st.stop()

    try:
        p = Path(bmce_path)
        if not p.exists():
            st.error(f"Local file not found: {bmce_path}")
            st.stop()

        if p.suffix.lower() == ".csv":
            preview_df = pd.read_csv(bmce_path)
        else:
            preview_df = pd.read_excel(bmce_path, engine="openpyxl")

        st.caption("File preview (first rows)")
        st.dataframe(preview_df.head(30), use_container_width=True)

    except Exception as e:
        st.error(f"Could not preview file: {e}")
        st.stop()

else:
    st.subheader("yfinance mode")
    st.caption("This requires `yfinance` in requirements.txt. BMCE is recommended for your desk data.")


# ============================================================
# Tabs: Backtest / Optimize
# ============================================================

tab_backtest, tab_opt = st.tabs(["Backtest", "Optimize"])

# ---- Global date range used by BOTH Backtest and Optimize ----
use_date_range = st.sidebar.checkbox("Use date range", value=False)

c1, c2 = st.sidebar.columns(2)
with c1:
    start_date = st.date_input("Start date", value=None, disabled=not use_date_range)
with c2:
    end_date = st.date_input("End date", value=None, disabled=not use_date_range)

start_str = start_date.isoformat() if (use_date_range and start_date) else None
end_str = end_date.isoformat() if (use_date_range and end_date) else None

if use_date_range and start_str and end_str and start_str > end_str:
    st.sidebar.error("Start date must be <= End date")
    st.stop()


# ============================================================
# Backtest Tab (FORM)
# ============================================================

with tab_backtest:
    st.subheader("Backtest")

    with st.form("backtest_form", clear_on_submit=False):

        st.markdown("### Strategy parameters")
        nan_policy = "flat"
        strategy_params: Dict[str, Any] = {}

        if strategy_kind == "ma_cross":
            fast = st.number_input("Fast SMA window", min_value=2, max_value=500, value=20, step=1)
            slow = st.number_input("Slow SMA window", min_value=3, max_value=500, value=50, step=1)
            if fast >= slow:
                st.warning("Fast must be < Slow. Auto-adjusting fast.")
                fast = min(int(fast), int(slow) - 1)
            strategy_params = {"sma_fast_window": int(fast), "sma_slow_window": int(slow), "allow_short": bool(allow_short), "nan_policy": nan_policy}
        elif strategy_kind == "sma_price":
            window = st.number_input("SMA window", min_value=2, max_value=500, value=50, step=1)
            strategy_params = {"sma_window": int(window), "allow_short": bool(allow_short), "nan_policy": nan_policy}
        elif strategy_kind == "rsi":
            window = st.number_input("RSI window", min_value=2, max_value=500, value=14, step=1)
            oversold = st.number_input("Oversold threshold", min_value=1, max_value=49, value=30, step=1)
            overbought = st.number_input("Overbought threshold", min_value=51, max_value=99, value=70, step=1)
            if oversold >= overbought:
                st.warning("Oversold must be < Overbought. Auto-adjusting oversold.")
                oversold = min(int(oversold), int(overbought) - 1)
            strategy_params = {
                "rsi_window": int(window),
                "rsi_oversold": int(oversold),
                "rsi_overbought": int(overbought),
                "allow_short": bool(allow_short),
                "nan_policy": nan_policy,
            }
        elif strategy_kind == "macd":
            fast = st.number_input("Fast EMA window", min_value=2, max_value=500, value=12, step=1)
            slow = st.number_input("Slow EMA window", min_value=3, max_value=500, value=26, step=1)
            signal = st.number_input("Signal EMA window", min_value=2, max_value=500, value=9, step=1)
            if fast >= slow:
                st.warning("Fast must be < Slow. Auto-adjusting fast.")
                fast = min(int(fast), int(slow) - 1)
            strategy_params = {
                "macd_fast_window": int(fast),
                "macd_slow_window": int(slow),
                "macd_signal_window": int(signal),
                "allow_short": bool(allow_short),
                "nan_policy": nan_policy,
            }
        elif strategy_kind == "bollinger":
            window = st.number_input("BB window", min_value=2, max_value=500, value=20, step=1)
            k = st.number_input("BB k (std dev multiplier)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
            strategy_params = {
                "bb_window": int(window),
                "bb_k": float(k),
                "allow_short": bool(allow_short),
                "nan_policy": nan_policy,
            }

        
        st.markdown("### Portfolio")
        initial_cash = st.number_input("Initial cash", min_value=1_000.0, value=100_000.0, step=10_000.0)
        rebalance_policy = st.selectbox("Rebalance policy", ["on_change", "every_bar"], index=0)
        sizing_mode = st.selectbox("Sizing mode", ["target_weight", "pct_cash_shares"], index=1)
        cooldown_bars = st.number_input("Min bars between trades (cooldown)", min_value=0, value=0, step=1)

        st.markdown("### Sizing (percentages)")
        buy_pct_cash = st.slider("Buy % of cash per entry", 0.01, 1.00, 0.25, 0.01)
        sell_pct_shares = st.slider("Sell % of shares per exit", 0.01, 1.00, 1.00, 0.01)
        min_return_before_sell = st.slider(
            "Min return before selling (take-profit, %)",
            0.0, 50.0, 0.0, 0.1
        ) / 100.0


        st.markdown("### Liquidity / Volume")

        use_volume_gate = st.checkbox("Enable volume gate (Layer 1)", value=False)
        volume_gate_kind = st.selectbox("Gate type", ["min_abs", "min_ratio_adv"], index=0, disabled=not use_volume_gate)

        min_volume_abs = st.number_input("Min Volume (abs shares)", min_value=0.0, value=0.0, step=10_000.0, disabled=(not use_volume_gate or volume_gate_kind != "min_abs"))
        min_volume_ratio_adv = st.slider("Min Volume / ADV ratio", 0.0, 2.0, 0.3, 0.05, disabled=(not use_volume_gate or volume_gate_kind != "min_ratio_adv"))
        volume_gate_adv_window = st.number_input("ADV window for gate", min_value=1, value=20, step=1, disabled=(not use_volume_gate or volume_gate_kind != "min_ratio_adv"))

        use_participation_cap = st.checkbox("Enable participation cap (Layer 3)", value=False)
        participation_rate = st.slider("Participation rate", 0.001, 0.50, 0.05, 0.001, disabled=not use_participation_cap)
        participation_basis = st.selectbox("Participation basis", ["bar", "adv"], index=0, disabled=not use_participation_cap)
        adv_window = st.number_input("ADV window (for cap)", min_value=1, value=20, step=1, disabled=(not use_participation_cap or participation_basis != "adv"))


        st.markdown("### Costs")
        apply_costs = st.checkbox("Apply costs", value=False)
        if apply_costs:
            brokerage_bps = st.number_input("Brokerage (bps)", value=0.0, step=1.0)
            comm_bourse_bps = st.number_input("Commission de la BdC (bps)", value=0.1, step=1.0)
            reg_liv_bps = st.number_input("Frais Règlement-Livraison (bps)", value=0.2, step=1.0)
            tva_rate = st.number_input("tva", value=0.0003, step=0.0001)
            slippage_bps = st.number_input("Slippage (bps)", value=0.0, step=1.0)
        else:
            brokerage_bps = comm_bourse_bps = reg_liv_bps = slippage_bps = 0.0
            tva_rate = 0.000

        st.markdown("### Strategy leaderboard (manual)")
        lb_enable = st.checkbox("Enable leaderboard run", value=False)

        lb_kinds = []
        lb_params_map = {}

        if lb_enable:
            lb_kinds = st.multiselect(
                "Select strategies to compare",
                options=["buy_hold", "ma_cross", "sma_price", "rsi", "macd", "bollinger"],
                default=["buy_hold", strategy_kind],
            )

            st.caption("Set strategy-specific params (baseline portfolio/cost/volume/date range is shared).")

            for k in lb_kinds:
                with st.expander(f"Params: {k}", expanded=(k != "buy_hold")):
                    lb_params_map[k] = render_strategy_params_ui(
                        k,
                        prefix=f"lb_{k}",
                        allow_short=allow_short,
                        nan_policy=nan_policy,
                    )

            rank_by = st.selectbox(
                "Rank by",
                options=["CAGR", "PnL", "Total Return", "Sharpe", "Max Drawdown", "Win %", "# Trades"],
                index=0,
                key="lb_rank_by",
            )

            lb_run = st.form_submit_button("Run leaderboard")
        else:
            lb_run = False



        run_backtest = st.form_submit_button("Run backtest")
    cost_model = CostModel(
            brokerage_bps=float(brokerage_bps),
            comm_bourse_bps=float(comm_bourse_bps),
            reg_liv_bps=float(reg_liv_bps),
            slippage_bps=float(slippage_bps),
            tva_rate=float(tva_rate),
        )
    tmp_path = None
    tmp_is_temp = True
    tmp_dir = None
    try:
        if source_key == "bmce":
            tmp_is_temp = False
            tmp_path = st.session_state.get("bmce_cached_path")
            if not tmp_path:
                st.error("BMCE file not persisted. Re-upload the file.")
                st.stop()

        if run_backtest:
            base_spec = make_base_spec(
                source_key=source_key,
                symbol=symbol,
                timezone=timezone,
                interval=interval,
                bmce_tmp_path=tmp_path,
                start=start_str,
                end=end_str,
                include_windows=include_windows,
                exclude_windows=exclude_windows,
                yf_period=yf_period,
                yf_interval=yf_interval,
                yf_auto_adjust=yf_auto_adjust,
                strategy_kind=strategy_kind,
                strategy_params=strategy_params,
                allow_short=bool(allow_short),
                initial_cash=float(initial_cash),
                rebalance_policy=str(rebalance_policy),
                sizing_mode=str(sizing_mode),
                buy_pct_cash=float(buy_pct_cash),
                sell_pct_shares=float(sell_pct_shares),
                cooldown_bars=int(cooldown_bars),
                cost_model=cost_model,
                use_volume_gate=use_volume_gate,
                volume_gate_kind=volume_gate_kind,
                min_volume_abs=min_volume_abs,
                min_volume_ratio_adv=min_volume_ratio_adv,
                volume_gate_adv_window=volume_gate_adv_window,
                use_participation_cap=use_participation_cap,
                participation_rate=participation_rate,
                participation_basis=participation_basis,
                adv_window=adv_window,
                min_return_before_sell=min_return_before_sell,
            )

            cmp_spec = None
            if compare_mode == "Buy & Hold (same data)":
                label_b = f"buy_hold | {symbol}"

                cmp_spec = EngineSpec(
                    data=base_spec.data,
                    indicators=base_spec.indicators,
                    strategy=StrategyConfig(kind="buy_hold", params={"buy_pct_cash": 1.0}),
                    portfolio=base_spec.portfolio,
                    benchmark=BenchmarkConfig(enabled=False),  # keep off
                    plot_indicators=[],
                    periods_per_year=base_spec.periods_per_year,
                    rf_annual=base_spec.rf_annual,
                )

            elif compare_mode == "Another strategy (same data)":
                cmp_spec = EngineSpec(
                    data=base_spec.data,
                    indicators=base_spec.indicators,
                    strategy=StrategyConfig(kind=compare_strategy_kind, params=compare_params),
                    portfolio=base_spec.portfolio,
                    benchmark=BenchmarkConfig(enabled=False),
                    plot_indicators=[],
                    periods_per_year=base_spec.periods_per_year,
                    rf_annual=base_spec.rf_annual,
                )

            # --- Run A (main strategy) once ---
            key_a = ("bundle", _spec_key(base_spec))
            bundle_a = st.session_state.get(key_a)
            if bundle_a is None:
                with st.spinner("Running backtest..."):
                    bundle_a = BacktestEngine(base_spec).run()
                st.session_state[key_a] = bundle_a

            # --- Run B (comparator) if enabled ---
            bundle_b = None
            if cmp_spec is not None:
                key_b = ("bundle", _spec_key(cmp_spec))
                bundle_b = st.session_state.get(key_b)
                if bundle_b is None:
                    with st.spinner("Running comparator..."):
                        bundle_b = BacktestEngine(cmp_spec).run()
                    st.session_state[key_b] = bundle_b

            # --- Attach comparator artifacts for display WITHOUT mutating cached bundle_a ---
            display_bundle_a = bundle_a  # default

            if bundle_b is not None:
                sym = symbol  # same data in your compare mode
                label_a = f"{base_spec.strategy.kind} | {sym}"
                label_b = f"{cmp_spec.strategy.kind} | {sym}"

                analyzer = ResultsAnalyzer(periods_per_year=base_spec.periods_per_year, rf_annual=base_spec.rf_annual)
                comp = analyzer.compare(
                    report_a=bundle_a.report,
                    report_b=bundle_b.report,
                    label_a=label_a,
                    label_b=label_b,
                )

                # Make a deep copy of the WHOLE bundle for display
                # (so we can mutate nested dicts safely)
                display_key = ("bundle_display", _spec_key(base_spec), _spec_key(cmp_spec))
                display_bundle_a = copy.deepcopy(bundle_a)

                

                # Now mutate the COPY's nested objects (allowed)
                rep0 = display_bundle_a.report

                # build new dicts without mutating frozen fields
                new_meta = dict(getattr(rep0, "meta", None) or {})
                new_meta["cum_compare_labels"] = {"a": label_a, "b": label_b}

                new_plots = dict(getattr(rep0, "plots", None) or {})
                new_plots["cum_vs_bench"] = comp["cum_plot"]

                new_tables = dict(getattr(rep0, "tables", None) or {})
                new_tables["curve_vs_comparator"] = comp["curve_table"]

                # create a NEW report object (works even if frozen)
                rep1 = replace(rep0, meta=new_meta, plots=new_plots, tables=new_tables)

                # now update the BUNDLE with the new report (bundle may be frozen too)
                try:
                    display_bundle_a = replace(display_bundle_a, report=rep1)
                    st.session_state[display_key] = display_bundle_a
                except Exception:
                    # if bundle is not a dataclass but a namedtuple or custom object:
                    if hasattr(display_bundle_a, "_replace"):
                        display_bundle_a = display_bundle_a._replace(report=rep1)
                    else:
                        # last resort: keep it in session and render using rep1 directly
                        # (but your render_bundle expects bundle.report, so this is rarely needed)
                        raise

                st.session_state[display_key] = display_bundle_a


            # --- Render ONLY ONCE: render the display bundle (copy if comparator on) ---
            render_bundle(display_bundle_a, port_cfg=base_spec.portfolio, label="",strategy_params=base_spec.strategy.params,)

        if lb_run:
            # 1) build baseline spec once using ONE "dummy" strategy (doesn't matter which)
            #    We'll immediately override strategy per loop.
            baseline_spec = make_base_spec(
                source_key=source_key,
                symbol=symbol,
                timezone=timezone,
                interval=interval,
                bmce_tmp_path=tmp_path,
                start=start_str,
                end=end_str,
                include_windows=include_windows,
                exclude_windows=exclude_windows,
                yf_period=yf_period,
                yf_interval=yf_interval,
                yf_auto_adjust=yf_auto_adjust,

                # dummy, overwritten below
                strategy_kind="buy_hold",
                strategy_params={"buy_pct_cash": 1.0},

                allow_short=bool(allow_short),
                initial_cash=float(initial_cash),
                rebalance_policy=str(rebalance_policy),
                sizing_mode=str(sizing_mode),
                buy_pct_cash=float(buy_pct_cash),
                sell_pct_shares=float(sell_pct_shares),
                cooldown_bars=int(cooldown_bars),
                cost_model=cost_model,
                use_volume_gate=use_volume_gate,
                volume_gate_kind=volume_gate_kind,
                min_volume_abs=min_volume_abs,
                min_volume_ratio_adv=min_volume_ratio_adv,
                volume_gate_adv_window=volume_gate_adv_window,
                use_participation_cap=use_participation_cap,
                participation_rate=participation_rate,
                participation_basis=participation_basis,
                adv_window=adv_window,
                min_return_before_sell=min_return_before_sell,
            )

            rows = []
            bundles = {}  # optional: store bundles for later drill-down

            with st.spinner("Running leaderboard..."):
                for k in lb_kinds:
                    params = lb_params_map.get(k, {}) or {}
                    spec_k = replace(
                        baseline_spec,
                        strategy=StrategyConfig(kind=k, params=params),
                        benchmark=BenchmarkConfig(enabled=False),
                        plot_indicators=[],
                    )

                    # cache like you already do
                    key_k = ("bundle", _spec_key(spec_k))
                    bundle_k = st.session_state.get(key_k)
                    if bundle_k is None:
                        bundle_k = BacktestEngine(spec_k).run()
                        st.session_state[key_k] = bundle_k

                    bundles[k] = bundle_k
                    name = f"{k}"
                    rows.append(leaderboard_row_from_report(bundle_k.report, name=name))

            df_lb = pd.DataFrame(rows)

            # 2) ranking logic
            # Higher is better except Max Drawdown (more negative is worse => rank by descending is wrong)
            # For Max Drawdown you want closer to 0 (i.e. higher value) or absolute?
            # We'll rank by "higher is better" for everything EXCEPT "# Trades" maybe neutral.
            # For Max Drawdown: higher (less negative) is better, so higher-is-better still works.
            sort_ascending = False
            if rank_by in ["# Trades"]:
                sort_ascending = False  # up to you
            df_lb = df_lb.sort_values(rank_by, ascending=sort_ascending, na_position="last").reset_index(drop=True)

            st.subheader("Strategy leaderboard")
            st.dataframe(df_lb, use_container_width=True)

            pick = st.selectbox("Drill into strategy", options=df_lb["Strategy"].tolist())
            pick_kind = pick.split()[0]  # if you used just k as Strategy name, it’s just pick
            render_bundle(bundles[pick_kind], port_cfg=baseline_spec.portfolio, label="",strategy_params=lb_params_map.get(pick_kind, {}),)
    finally:
        if tmp_is_temp and tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        if tmp_dir and os.path.isdir(tmp_dir):
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass



# ============================================================
# Optimize Tab (FORM + interval editors)
# ============================================================

with tab_opt:
    st.subheader("Optimize (rank by pnl then cagr)")
    # Baseline fixed values (used when NOT optimized)
    with st.expander("Fixed baseline (used when NOT optimized)", expanded=False):
        initial_cash0 = st.number_input("Initial cash (baseline)", min_value=1_000.0, value=100_000.0, step=10_000.0, key="opt_initial_cash")
        rebalance_policy0 = st.selectbox("Rebalance policy (baseline)", ["on_change", "every_bar"], index=0, key="opt_reb_policy")
        sizing_mode0 = st.selectbox("Sizing mode (baseline)", ["target_weight", "pct_cash_shares"], index=1, key="opt_sizing_mode")
        cooldown0 = st.number_input("cooldown_bars (baseline)", min_value=0, value=0, step=1, key="opt_cd0")
        buy0 = st.slider("buy_pct_cash (baseline)", 0.01, 1.00, 0.25, 0.01, key="opt_buy0")
        sell0 = st.slider("sell_pct_shares (baseline)", 0.01, 1.00, 1.00, 0.01, key="opt_sell0")

        nan_policy0 = "flat"
        st.markdown("### Strategy parameters (baseline)")
        if strategy_kind == "ma_cross":
            fast0 = st.number_input("fast_window (baseline)", min_value=2, max_value=500, value=20, step=1, key="opt_fast0")
            slow0 = st.number_input("slow_window (baseline)", min_value=3, max_value=500, value=50, step=1, key="opt_slow0")
            if fast0 >= slow0:
                st.warning("Baseline fast must be < slow. Auto-adjusting.")
                fast0 = min(int(fast0), int(slow0) - 1)
            strategy_params0 = {"sma_fast_window": int(fast0), "sma_slow_window": int(slow0), "allow_short": bool(allow_short), "nan_policy": nan_policy0}
        elif strategy_kind == "sma_price":
            w0 = st.number_input("window (baseline)", min_value=2, max_value=500, value=50, step=1, key="opt_w0")
            strategy_params0 = {"sma_window": int(w0), "allow_short": bool(allow_short), "nan_policy": nan_policy0}
        elif strategy_kind == "rsi":
            period = st.number_input("RSI period", min_value=2, max_value=200, value=14, step=1)
            low = st.number_input("RSI low", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
            high = st.number_input("RSI high", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
            mode = st.selectbox("RSI mode", ["reversal", "momentum"], index=0)
            strategy_params0 = {"rsi_window": int(period), "rsi_oversold": float(low), "rsi_overbought": float(high), "mode": mode, "allow_short": bool(allow_short), "nan_policy": nan_policy0}

        elif strategy_kind == "macd":
            fast = st.number_input("MACD fast", min_value=2, max_value=200, value=12, step=1)
            slow = st.number_input("MACD slow", min_value=3, max_value=400, value=26, step=1)
            if fast >= slow:
                st.warning("Fast must be < Slow. Auto-adjusting fast.")
                fast = min(int(fast), int(slow) - 1)
            sig = st.number_input("MACD signal", min_value=2, max_value=200, value=9, step=1)
            trigger = st.selectbox("MACD trigger", ["cross", "zero"], index=0)
            strategy_params0 = {"macd_fast_window": int(fast), "macd_slow_window": int(slow), "macd_signal_window": int(sig), "trigger": trigger, "allow_short": bool(allow_short), "nan_policy": nan_policy0}
        elif strategy_kind == "bollinger":
            window = st.number_input("BB window (baseline)", min_value=2, max_value=500, value=20, step=1, key="opt_bb_window0")
            k = st.number_input("BB k (std dev multiplier) (baseline)", min_value=0.1, max_value=5.0, value=2.0, step=0.1, key="opt_bb_k0")
            strategy_params0 = {"bb_window": int(window), "bb_k": float(k), "allow_short": bool(allow_short), "nan_policy": nan_policy0}

        min_ret0 = st.slider("Min return before sell baseline (%)", 0.0, 50.0, 0.0, 0.1, key="opt_min_ret0") / 100.0

        st.markdown("### Liquidity / Volume (baseline)")
        use_volume_gate0 = st.checkbox("Enable volume gate (baseline)", value=False, key="opt_use_volume_gate")
        volume_gate_kind0 = st.selectbox(
            "Gate type (baseline)",
            ["min_abs", "min_ratio_adv"],
            index=0,
            disabled=not use_volume_gate0,
            key="opt_volume_gate_kind",
        )

        min_volume_abs0 = st.number_input(
            "Min Volume (abs shares) baseline",
            min_value=0.0,
            value=0.0,
            step=10_000.0,
            disabled=(not use_volume_gate0 or volume_gate_kind0 != "min_abs"),
            key="opt_min_volume_abs",
        )
        min_volume_ratio_adv0 = st.slider(
            "Min Volume / ADV ratio baseline",
            0.0, 2.0, 0.3, 0.05,
            disabled=(not use_volume_gate0 or volume_gate_kind0 != "min_ratio_adv"),
            key="opt_min_volume_ratio_adv",
        )
        volume_gate_adv_window0 = st.number_input(
            "ADV window for gate baseline",
            min_value=1,
            value=20,
            step=1,
            disabled=(not use_volume_gate0 or volume_gate_kind0 != "min_ratio_adv"),
            key="opt_volume_gate_adv_window",
        )
        use_participation_cap0 = st.checkbox("Enable participation cap (baseline)", value=False, key="opt_use_participation_cap")
        participation_rate0 = st.slider(
            "Participation rate baseline",
            0.001, 0.50, 0.05, 0.001,
            disabled=not use_participation_cap0,
            key="opt_participation_rate",
        )
        participation_basis0 = st.selectbox(
            "Participation basis baseline",
            ["bar", "adv"],
            index=0,
            disabled=not use_participation_cap0,
            key="opt_participation_basis",
        )
        adv_window0 = st.number_input(
            "ADV window (for cap) baseline",
            min_value=1,
            value=20,
            step=1,
            disabled=(not use_participation_cap0 or participation_basis0 != "adv"),
            key="opt_adv_window",
        )


        st.markdown("### Costs (baseline)")
        apply_costs0 = st.checkbox("Apply costs", value=False, key="opt_apply_costs")
        if apply_costs0:
            brokerage_bps0 = st.number_input("Brokerage (bps)", value=0.0, step=1.0, key="opt_brok")
            comm_bourse_bps0 = st.number_input("Commission de la BdC (bps)", value=0.1, step=1.0, key="opt_exch")
            reg_liv_bps0 = st.number_input("Frais Règlement-Livraison (bps)", value=0.2, step=1.0, key="opt_settle")
            tva_rate0 = st.number_input("tva", value=0.003, step=0.001, key="opt_tva")
            slippage_bps0 = st.number_input("Slippage (bps)", value=0.0, step=1.0, key="opt_slip")
        else:
            brokerage_bps0 = comm_bourse_bps0 = reg_liv_bps0 = slippage_bps0 = 0.0
            tva_rate0 = 0.000

    # Build catalog for THIS strategy_kind
    catalog = default_param_catalog(strategy_kind)
    selectable_keys = list(catalog.keys())

    default_active = (
        ["strategy.sma_fast_window", "strategy.sma_slow_window"] if strategy_kind == "ma_cross"
        else ["strategy.sma_window"] if strategy_kind == "sma_price"
        else ["strategy.rsi_window", "strategy.rsi_oversold", "strategy.rsi_overbought"] if strategy_kind == "rsi"
        else ["strategy.macd_fast_window", "strategy.macd_slow_window", "strategy.macd_signal_window"] if strategy_kind == "macd"
        else ["strategy.bb_window", "strategy.bb_k"] if strategy_kind == "bollinger"
        else []
    )

    active_keys = st.multiselect(
        "Select parameters to optimize",
        options=selectable_keys,
        default=default_active,
        key="active_keys",
    )


    # Interval editors (IMPORTANT: ParamDef is frozen -> use replace)
    st.markdown("### Intervals / choices for selected parameters")
    edited_catalog: Dict[str, ParamDef] = dict(catalog)

    for k in active_keys:
        pdef = edited_catalog[k]
        st.markdown(f"**{k}**  (`{pdef.kind}`)")

        if pdef.kind == "int":
            lo, hi, step = pdef.domain

            mode = st.radio(
                "Domain mode",
                ["range", "manual list"],
                index=0,
                horizontal=True,
                key=f"{k}_mode",
            )

            if mode == "range":
                c1, c2, c3 = st.columns(3)
                lo2 = c1.number_input(f"{k} min", value=int(lo), step=1, key=f"{k}_min")
                hi2 = c2.number_input(f"{k} max", value=int(hi), step=1, key=f"{k}_max")
                step2 = c3.number_input(f"{k} step", value=int(step), step=1, key=f"{k}_step")
                if lo2 > hi2:
                    st.error(f"{k}: min must be <= max")
                    st.stop()
                edited_catalog[k] = replace(pdef, domain=(int(lo2), int(hi2), int(step2)))

            else:
                # manual list -> we convert to a CHOICE domain so optimizer uses exactly those values
                default_txt = default_manual_txt_for_param(k, pdef)
                state_key = f"opt_{strategy_kind}_{k}_manual"

                txt = st.text_input(
                    f"{k} values (comma/space separated)",
                    value=st.session_state.get(state_key, default_txt),
                    key=state_key,
                )


                try:
                    vals = parse_int_list(txt)
                except Exception as e:
                    st.error(f"{k}: invalid list: {e}")
                    st.stop()

                if not vals:
                    st.error(f"{k}: please provide at least one value.")
                    st.stop()

                edited_catalog[k] = replace(pdef, kind="choice", domain=vals)


        elif pdef.kind == "float":
            lo, hi, step = pdef.domain
            c1, c2, c3 = st.columns(3)
            lo2 = c1.number_input(f"{k} min", value=float(lo), step=0.01, key=f"{k}_min")
            hi2 = c2.number_input(f"{k} max", value=float(hi), step=0.01, key=f"{k}_max")
            step2 = c3.number_input(f"{k} step", value=float(step), step=0.01, key=f"{k}_step")
            if lo2 > hi2:
                st.error(f"{k}: min must be <= max")
                st.stop()
            edited_catalog[k] = replace(pdef, domain=(float(lo2), float(hi2), float(step2)))

        elif pdef.kind == "choice":
            choices = list(pdef.domain)
            picked = st.multiselect(f"{k} choices", choices, default=choices, key=f"{k}_choices")
            edited_catalog[k] = replace(pdef, domain=list(picked))

        elif pdef.kind == "date_window":
            if source_key != "bmce":
                st.warning("data.window optimization is intended for BMCE uploads.")
                edited_catalog[k] = replace(pdef, domain=[])
            else:
                min_bars = st.number_input("min bars per window", min_value=30, value=252, step=21, key="dw_min_bars")
                step_bars = st.number_input("step bars", min_value=1, value=21, step=1, key="dw_step_bars")
                max_windows = st.number_input("max windows", min_value=10, value=200, step=10, key="dw_max_windows")
                # domain filled at run time after preview_df exists

    if "opt_catalog_by_kind" not in st.session_state:
        st.session_state["opt_catalog_by_kind"] = {}
    if "opt_active_keys_by_kind" not in st.session_state:
        st.session_state["opt_active_keys_by_kind"] = {}

    # Save current strategy_kind edits so they can be reused later in optimize-all
    st.session_state["opt_catalog_by_kind"][strategy_kind] = edited_catalog
    st.session_state["opt_active_keys_by_kind"][strategy_kind] = list(active_keys)


    st.markdown("### Optimization method")
    method = st.selectbox("Method", ["random", "grid"], index=0, key="opt_method")
    top_k = st.number_input("Show top K", min_value=5, value=30, step=5, key="opt_topk")

    if method == "random":
        n_trials = st.number_input("Trials", min_value=10, value=200, step=10, key="opt_trials")
    else:
        n_trials = 0

    BATCH_PERIODS = dict(MASI_PERIODS)  # start with MASI
    if symbol.upper() == "IAM":
        # namespace IAM keys to prevent accidental collisions
        BATCH_PERIODS.update({f"IAM — {k}": v for k, v in IAM_PERIODS.items()})


    st.subheader("Batch tests")

    do_batch = st.checkbox("Batch test by period", value=False)

    selected_periods = []
    if do_batch:
        period_options = list(BATCH_PERIODS.keys())
        selected_periods = st.multiselect(
            "Select periods to run",
            options=period_options,
            default=period_options,
        )

        batch_objective = st.selectbox(
            "Objective for comparison",
            ["pnl", "cagr"],  # match what stats dict returns
            index=0
        )

        optimize_within_each = st.checkbox("Optimize within each period", value=True)


    run_opt = st.button("Run optimization", key="run_opt_btn")

    st.markdown("### Optimization scope")
    opt_scope = st.radio(
        "What do you want to optimize?",
        ["This strategy only", "All selected strategies (leaderboard)"],
        index=0,
        horizontal=True,
        key="opt_scope",
    )

    rank_metric = st.selectbox(
        "Leaderboard ranking metric",
        ["CAGR", "PnL", "Total Return", "Sharpe", "Max Drawdown", "Win %", "# Trades"],
        index=0,
        key="opt_lb_rank_metric",
    )

    if opt_scope == "All selected strategies (leaderboard)":
        ALL_KINDS = ["buy_hold", "ma_cross", "sma_price", "rsi", "macd", "bollinger"]
        lb_opt_kinds = st.multiselect(
            "Strategies to optimize/compare",
            options=ALL_KINDS,
            default=["buy_hold", "ma_cross", "sma_price", "rsi", "macd", "bollinger"],
            key="opt_lb_kinds",
        )
        st.markdown("### Per-strategy optimization domains")
        st.caption("Configure which parameters to optimize and their intervals for EACH strategy in the leaderboard.")

        for k in lb_opt_kinds:
            if k == "buy_hold":
                continue
            with st.expander(f"Optimize domains: {k}", expanded=False):
                render_interval_editor_for_kind(k, source_key=source_key, preview_df=preview_df)
        st.divider()
        st.subheader("Workbook batch (Optimize leaderboard export)")

        can_run_all = (source_key == "bmce") and (tmp_path is not None) and str(tmp_path).lower().endswith((".xlsx", ".xls"))

        if not can_run_all:
            st.caption("To run across all tickers, use BMCE Excel workbook (multi-sheet).")

        run_all_sheets = st.button(
            "Run OPTIMIZED leaderboard for ALL sheets + export ZIP",
            disabled=not can_run_all,
            key="run_opt_lb_all_sheets",
        )
        
        # list sheet names (tickers)
        all_sheets = _get_excel_sheets(tmp_path)

        # optional: let user choose subset (keeps it practical)
        selected_sheets = st.multiselect(
            "Select tickers (sheets) to run",
            options=all_sheets,
            default=all_sheets,
            key="opt_lb_all_sheets_select",
        )

        if run_all_sheets:
            if not all_sheets:
                st.error("No sheets found in workbook.")
                st.stop()

            if not selected_sheets:
                st.warning("No tickers selected.")
                st.stop()

            # safety: date_window optimization per sheet requires per-sheet preview_df.
            # Keep it simple: disallow date_window in batch unless you explicitly add per-sheet preview.
            for kk in lb_opt_kinds:
                if kk == "buy_hold":
                    continue
                act = st.session_state.get("opt_active_keys_by_kind", {}).get(kk, [])
                if "data.window" in (act or []):
                    st.error("Workbook batch does not support optimizing data.window across sheets (needs per-sheet windows). Disable data.window and retry.")
                    st.stop()

            _ensure_site_store()
            store = st.session_state["site_export_store"]

            prog = st.progress(0)
            status = st.empty()

            # We reuse your existing base_spec (built inside run_opt), but for workbook batch we need one.
            # Build a base_spec using the baseline fields currently selected in the Optimize expander.
            cost_model = CostModel(
                brokerage_bps=float(brokerage_bps0),
                comm_bourse_bps=float(comm_bourse_bps0),
                reg_liv_bps=float(reg_liv_bps0),
                slippage_bps=float(slippage_bps0),
                tva_rate=float(tva_rate0),
            )

            base_spec_batch = make_base_spec(
                source_key=source_key,
                symbol=selected_sheets[0],  # placeholder; overwritten per sheet in helper
                timezone=timezone,
                interval=interval,
                bmce_tmp_path=tmp_path,
                start=start_str,
                end=end_str,
                include_windows=include_windows,
                exclude_windows=exclude_windows,
                yf_period=yf_period,
                yf_interval=yf_interval,
                yf_auto_adjust=yf_auto_adjust,
                strategy_kind=strategy_kind,          # doesn't matter; helper replaces
                strategy_params=strategy_params0,     # baseline dict, ok
                allow_short=bool(allow_short),
                initial_cash=float(initial_cash0),
                rebalance_policy=str(rebalance_policy0),
                sizing_mode=str(sizing_mode0),
                buy_pct_cash=float(buy0),
                sell_pct_shares=float(sell0),
                cooldown_bars=int(cooldown0),
                cost_model=cost_model,
                use_volume_gate=bool(use_volume_gate0),
                volume_gate_kind=str(volume_gate_kind0),
                min_volume_abs=float(min_volume_abs0),
                min_volume_ratio_adv=float(min_volume_ratio_adv0),
                volume_gate_adv_window=int(volume_gate_adv_window0),
                use_participation_cap=bool(use_participation_cap0),
                participation_rate=float(participation_rate0),
                participation_basis=str(participation_basis0),
                adv_window=int(adv_window0),
                min_return_before_sell=float(min_ret0),
            )

            # opt_cfg is already defined above; if not, rebuild it:
            opt_cfg_batch = OptimizeConfig(
                method=str(method),
                seed=42,
                n_trials=int(n_trials) if method == "random" else 0,
                top_k=int(top_k),
            )

            # ---- Build domains_by_kind (MVP: capture active keys per kind) ----
            domains_by_kind = {}
            for kk in lb_opt_kinds:
                if kk == "buy_hold":
                    continue
                domains_by_kind[kk] = st.session_state.get("opt_active_keys_by_kind", {}).get(kk, [])

            cost_model_spec = {
                "brokerage_bps": float(brokerage_bps0),
                "comm_bourse_bps": float(comm_bourse_bps0),
                "reg_liv_bps": float(reg_liv_bps0),
                "slippage_bps": float(slippage_bps0),
                "tva_rate": float(tva_rate0),
            }
            volume_gate_spec = {
                "enabled": bool(use_volume_gate0),
                "kind": str(volume_gate_kind0),
                "min_volume_abs": float(min_volume_abs0),
                "min_volume_ratio_adv": float(min_volume_ratio_adv0),
                "adv_window": int(volume_gate_adv_window0),
            }
            participation_cap_spec = {
                "enabled": bool(use_participation_cap0),
                "rate": float(participation_rate0),
                "basis": str(participation_basis0),
                "adv_window": int(adv_window0),
            }

            run_spec = build_run_spec(
                source_key=source_key,
                symbols=list(selected_sheets),
                start=start_str,
                end=end_str,
                include_windows=include_windows,
                exclude_windows=exclude_windows,
                interval=str(interval),
                yf_period=str(yf_period),
                yf_interval=str(yf_interval),
                yf_auto_adjust=bool(yf_auto_adjust),
                rank_metric=str(rank_metric),
                lb_opt_kinds=list(lb_opt_kinds),
                opt_method=str(method),
                n_trials=int(n_trials) if method == "random" else 0,
                top_k=int(top_k),
                allow_short=bool(allow_short),
                initial_cash=float(initial_cash0),
                cooldown_bars=int(cooldown0),
                min_return_before_sell=float(min_ret0),
                cost_model=cost_model_spec,
                volume_gate=volume_gate_spec,
                participation_cap=participation_cap_spec,
                domains_by_kind=domains_by_kind,
                app_version="v1",
            )

            run_id = run_id_from_spec(run_spec)
            run_dir = RESULTS_ROOT / "runs" / run_id
            done_flag = run_dir / "DONE"

            st.info(f"Run ID: `{run_id}`")

            if done_flag.exists():
                st.success("Cached run found on disk — skipping optimization.")
                st.caption(f"Run folder: {run_dir.as_posix()}")

                # If the site artifacts are missing/incomplete, rebuild them from the current session store.
                if not _run_folder_has_site(run_dir):
                    store_now = st.session_state.get("site_export_store", {})
                    if store_now:
                        st.warning("Cached run is missing site artifacts (plots/ledgers). Rebuilding export from session store...")
                        zip_site = _make_site_zip_from_store(
                            store=store_now,
                            site_title="Backtest Results",
                        )
                        _extract_zip_bytes_to_folder(zip_site, run_dir)
                        st.success("Rebuilt site artifacts into the run folder.")
                    else:
                        # No store in memory => you cannot recreate plots/ledgers deterministically.
                        st.error(
                            "Cached run exists (DONE) but the run folder has no site artifacts, "
                            "and the session store is empty. "
                            "Delete the DONE flag (or add a 'force rerun' button) to recompute/export."
                        )
                        st.code(str(done_flag), language="text")
                        st.stop()

                # viewer link
                st.code(f"results_website/index.html?run_id={run_id}", language="text")

                # Always offer ZIP download if we can (from disk or rebuilt)
                # simplest: re-zip run_dir for download OR regenerate from store if you prefer
                if st.session_state.get("site_export_store"):
                    zip_site = _make_site_zip_from_store(
                        store=st.session_state["site_export_store"],
                        site_title="Backtest Results",
                    )
                    _extract_zip_bytes_to_folder(zip_site, run_dir)

                    st.download_button(
                        label="Download RESULTS WEBSITE ZIP (ALL tickers)",
                        data=zip_site,
                        file_name="results_website.zip",
                        mime="application/zip",
                        key="dl_site_zip_all_tickers_cached",
                    )

                st.stop()




            for i, sym_i in enumerate(selected_sheets, start=1):
                status.write(f"Running {sym_i} ({i}/{len(selected_sheets)}) ...")

                df_lb_i, best_bundles_i, best_specs_i, top5_names_i = run_opt_leaderboard_for_one_symbol(
                    symbol_i=sym_i,
                    base_spec=base_spec_batch,
                    lb_opt_kinds=lb_opt_kinds,
                    opt_cfg=opt_cfg_batch,
                    rank_metric=rank_metric,
                )

                if df_lb_i is None or df_lb_i.empty:
                    status.write(f"Skipped {sym_i}: no results.")
                    prog.progress(i / len(selected_sheets))
                    continue

                _push_ticker_results_to_site_store(
                    ticker=sym_i,
                    leaderboard_df=df_lb_i,
                    best_bundles=best_bundles_i,
                    best_specs=best_specs_i,
                    top5_names=top5_names_i,
                    title="Backtest Results",
                )

                prog.progress(i / len(selected_sheets))

            # build one website ZIP for all saved tickers
            zip_site = _make_site_zip_from_store(
                store=st.session_state["site_export_store"],
                site_title="Backtest Results",
            )
            _extract_zip_bytes_to_folder(zip_site, run_dir)


            st.success(f"Done. Export includes: {', '.join(sorted(st.session_state['site_export_store'].keys()))}")
            st.download_button(
                label="Download RESULTS WEBSITE ZIP (ALL tickers)",
                data=zip_site,
                file_name="results_website.zip",
                mime="application/zip",
                key="dl_site_zip_all_tickers",
            )

            # after the loop when store is complete:
            store = st.session_state.get("site_export_store", {})

            run_dir.mkdir(parents=True, exist_ok=True)

            export_run_to_results_site(
                store=store,
                run_id=run_id,
                run_spec=run_spec,
                run_dir=run_dir,
                results_root=RESULTS_ROOT,
            )
            # mark run complete for disk cache short-circuit
            done_flag.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


            st.success("Run exported to results_website.")
            st.caption(f"Run folder: {run_dir.as_posix()}")

            # Optional: show local “viewer” link pattern
            st.code(f"results_website/index.html?run_id={run_id}", language="text")


            st.stop()





    if run_opt:
        tmp_path = None
        tmp_dir = None
        tmp_is_temp = True
        bench_tmp_path = None
        bench_tmp_dir = None
        bench_is_temp = True
        try:
            if use_benchmark and bench_source_key == "bmce":
                if bench_bmce_file is None:
                    st.warning("Benchmark enabled but no BMCE benchmark file uploaded.")
                    st.stop()

                suffix_b = Path(bench_bmce_file.name).suffix.lower()
                bench_is_temp = False
                bench_tmp_path = st.session_state.get("bench_cached_path")
                if not bench_tmp_path:
                    st.error("Benchmark BMCE file not persisted. Re-upload the benchmark file.")
                    st.stop()
            if source_key == "bmce":
                # Use stable persisted path (content-hash) to enable caching / speed
                tmp_is_temp = False
                tmp_path = st.session_state.get("bmce_cached_path")
                if not tmp_path:
                    st.error("BMCE file not persisted. Re-upload the file.")
                    st.stop()

            # Fill date windows domain if selected
            if "data.window" in active_keys:
                windows = build_date_windows_from_df(
                    preview_df,
                    min_bars=int(st.session_state.get("dw_min_bars", 252)),
                    step_bars=int(st.session_state.get("dw_step_bars", 21)),
                    max_windows=int(st.session_state.get("dw_max_windows", 200)),
                )
                edited_catalog["data.window"] = replace(edited_catalog["data.window"], domain=windows)

            # Build active_params list for optimizer (this is what run_optimization expects)
            active_params: List[ParamDef] = []
            for k in active_keys:
                p = edited_catalog[k]
                if p.enabled and p.domain is not None and (p.kind != "date_window" or len(p.domain) > 0):
                    active_params.append(p)

            if not active_params:
                st.error("No active parameters to optimize (empty domains).")
                st.stop()

            cost_model = CostModel(
                brokerage_bps=float(brokerage_bps0),
                comm_bourse_bps=float(comm_bourse_bps0),
                reg_liv_bps=float(reg_liv_bps0),
                slippage_bps=float(slippage_bps0),
                tva_rate=float(tva_rate0),
            )

            base_spec = make_base_spec(
                source_key=source_key,
                symbol=symbol,
                timezone=timezone,
                interval=interval,
                bmce_tmp_path=tmp_path,
                start=start_str,
                end=end_str,
                include_windows=include_windows,
                exclude_windows=exclude_windows,
                yf_period=yf_period,
                yf_interval=yf_interval,
                yf_auto_adjust=yf_auto_adjust,
                strategy_kind=strategy_kind,
                strategy_params=strategy_params0,
                allow_short=bool(allow_short),
                initial_cash=float(initial_cash0),
                rebalance_policy=str(rebalance_policy0),
                sizing_mode=str(sizing_mode0),
                buy_pct_cash=float(buy0),
                sell_pct_shares=float(sell0),
                cooldown_bars=int(cooldown0),
                use_volume_gate=bool(use_volume_gate0),
                volume_gate_kind=str(volume_gate_kind0),
                min_volume_abs=float(min_volume_abs0),
                min_volume_ratio_adv=float(min_volume_ratio_adv0),
                volume_gate_adv_window=int(volume_gate_adv_window0),

                use_participation_cap=bool(use_participation_cap0),
                participation_rate=float(participation_rate0),
                participation_basis=str(participation_basis0),
                adv_window=int(adv_window0),
                cost_model=cost_model,
                min_return_before_sell=float(min_ret0),
            )

            opt_cfg = OptimizeConfig(
                method=str(method),
                seed=42,
                n_trials=int(n_trials) if method == "random" else 0,
                top_k=int(top_k),
                # NOTE: no "objective" field in your OptimizeConfig
            )
            if opt_scope == "All selected strategies (leaderboard)":
                rows = []
                best_specs: dict[str, EngineSpec] = {}
                best_bundles: dict[str, Any] = {}

                # Common baseline spec (same data/portfolio/cost/volume/date range)
                base_spec_common = replace(
                    base_spec,
                    strategy=StrategyConfig(kind="buy_hold", params={"buy_pct_cash": 1.0}),
                    benchmark=BenchmarkConfig(enabled=False),
                    plot_indicators=[],
                )

                # 1) Optional Buy&Hold row (no optimization)
                bh_bundle = None
                if "buy_hold" in lb_opt_kinds:
                    with st.spinner("Backtesting buy_hold..."):
                        bh_bundle = BacktestEngine(base_spec_common).run()
                    best_specs["buy_hold"] = base_spec_common
                    best_bundles["buy_hold"] = bh_bundle

                    row = leaderboard_row_from_report(bh_bundle.report, name="buy_hold")
                    row["Best Params"] = format_params_for_table(base_spec_common.strategy.params)
                    rows.append(row)

                # 2) Optimize each selected strategy kind
                for k in lb_opt_kinds:
                    if k == "buy_hold":
                        continue

                    active_params_k = build_active_params_for_kind(k)
                    if not active_params_k:
                        st.warning(f"{k}: no active params. Skipping.")
                        continue

                    # Important: give the strategy some baseline params if you have them
                    # For now {} is okay if your strategy has defaults internally.
                    base_spec_k = replace(
                        base_spec_common,
                        strategy=StrategyConfig(kind=k, params={}),
                        benchmark=BenchmarkConfig(enabled=False),
                        plot_indicators=[],
                    )

                    with st.spinner(f"Optimizing {k}..."):
                        best_k, top_df_k, best_params_k, best_spec_k, ranked_df_k = run_optimization(
                            base_spec=base_spec_k,
                            active_params=active_params_k,
                            cfg=opt_cfg,
                        )

                    # Backtest best
                    with st.spinner(f"Backtesting best {k}..."):
                        # OPTIONAL: cache best bundle
                        key_best = ("bundle", _spec_key(best_spec_k))
                        bundle_k = st.session_state.get(key_best)
                        if bundle_k is None:
                            bundle_k = BacktestEngine(best_spec_k).run()
                            st.session_state[key_best] = bundle_k

                    best_specs[k] = best_spec_k
                    best_bundles[k] = bundle_k

                    row = leaderboard_row_from_report(bundle_k.report, name=f"{k} (best)")

                    # --- compute last-bar signal for the best strategy ---
                    sym0 = bundle_k.md.symbols()[0]
                    bars_k = bundle_k.md.bars[sym0]
                    pp_k = bundle_k.report.plots.get("price_panel", {})
                    ind_k = pp_k.get("indicators")

                    sig_date, sig_num, sig_lab = latest_signal_from_kind(
                        kind=k,
                        bars=bars_k,
                        indicators=ind_k,
                        params=best_spec_k.strategy.params,
                    )

                    row["Signal Date"] = sig_date.date().isoformat()
                    row["Signal"] = sig_lab
                    row["Signal (-1/0/+1)"] = int(sig_num)

                    row["Best Params"] = format_params_for_table(best_params_snapshot(best_spec_k))
                    rows.append(row)


                df_lb = pd.DataFrame(rows)

                # Sort: higher is better for your metrics (Max Drawdown is negative, closer to 0 = higher = better)
                df_lb = df_lb.sort_values(rank_metric, ascending=False, na_position="last").reset_index(drop=True)

                top5_df = df_lb.head(5).copy()

                st.subheader("Optimized strategies leaderboard")
                front = [c for c in ["Strategy", "Signal", "Signal Date", "CAGR", "PnL"] if c in df_lb.columns]
                rest = [c for c in df_lb.columns if c not in front]
                df_lb = df_lb[front + rest]

                st.dataframe(df_lb, use_container_width=True)

                st.subheader("Top 5 — Price + Indicators + Trades")
                top5_names = top5_df["Strategy"].tolist()
                tabs = st.tabs(top5_names)

                for i, strat_name in enumerate(top5_names):
                    kind = strat_name.replace(" (best)", "")
                    if kind not in best_bundles or kind not in best_specs:
                        continue

                    b = best_bundles[kind]
                    spec = best_specs[kind]

                    with tabs[i]:
                        st.markdown(f"### {strat_name}")
                        st.json(spec.strategy.params)

                        sym0 = b.md.symbols()[0]
                        bars = b.md.bars[sym0]
                        pp = b.report.plots.get("price_panel", {})

                        fig = plot_price_indicators_trades_line(
                            bars=bars,
                            strategy_params=spec.strategy.params,
                            indicators=pp.get("indicators"),
                            trades=pp.get("trades"),
                            indicator_cols=pp.get("indicator_cols"),
                            port_cfg=spec.portfolio,
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Export (needs kaleido)
                        try:
                            png_bytes = fig.to_image(format="png", width=2400, height=1350, scale=2)
                            st.download_button(
                                label="Download plot (PNG, high-res)",
                                data=png_bytes,
                                file_name=f"{kind}_price_panel.png",
                                mime="image/png",
                                key=f"dl_png_{kind}",
                            )
                        except Exception as e:
                            st.warning(f"PNG export unavailable (install kaleido): {e}")

                        try:
                            svg_bytes = fig.to_image(format="svg", width=2400, height=1350)
                            st.download_button(
                                label="Download plot (SVG, vector)",
                                data=svg_bytes,
                                file_name=f"{kind}_price_panel.svg",
                                mime="image/svg+xml",
                                key=f"dl_svg_{kind}",
                            )
                        except Exception as e:
                            st.warning(f"SVG export unavailable (install kaleido): {e}")
                # --- One-click download: leaderboard + top5 plots ---
                top5_kinds = [n.replace(" (best)", "") for n in top5_names]  # your naming convention
                # --- Save this ticker's results into a multi-ticker store (session) ---
                _push_ticker_results_to_site_store(
                    ticker=symbol,                      # IMPORTANT: current selected ticker/sheet
                    leaderboard_df=df_lb,
                    best_bundles=best_bundles,
                    best_specs=best_specs,
                    top5_names=top5_names,
                    title="Backtest Results",
                )

                zip_bytes = _make_leaderboard_zip(
                    leaderboard_df=df_lb,
                    top5_kinds=top5_kinds,
                    best_bundles=best_bundles,
                    best_specs=best_specs,
                )

                st.divider()
                st.subheader("Website export (multi-stock)")

                _ensure_site_store()
                store = st.session_state["site_export_store"]

                zip_site = _make_site_zip_from_store(
                    store=store,
                    site_title="Backtest Results",
                )

                st.download_button(
                    label="Download RESULTS WEBSITE ZIP (all saved tickers)",
                    data=zip_site,
                    file_name="results_website.zip",
                    mime="application/zip",
                    key="dl_site_zip",
                )

                st.caption(f"Saved tickers in this session: {', '.join(sorted(store.keys())) if store else '(none)'}")

                if st.button("Clear saved website results", key="clear_site_store"):
                    st.session_state["site_export_store"] = {}
                    st.success("Cleared.")


                # Persist for reruns
                st.session_state["opt_leaderboard_df"] = df_lb
                st.session_state["opt_leaderboard_bundles"] = best_bundles
                st.session_state["opt_leaderboard_specs"] = best_specs

                st.stop()

            if do_batch and selected_periods:
                df_batch = batch_optimize_by_period(
                    base_spec=base_spec,
                    active_params=active_params,
                    cfg=opt_cfg,
                    periods=BATCH_PERIODS,
                    selected_period_labels=selected_periods,
                    objective=batch_objective,
                )

                df_batch_sorted = df_batch.sort_values("objective_value", ascending=False).reset_index(drop=True)
                st.dataframe(df_batch_sorted, use_container_width=True)

                if df_batch_sorted.empty:
                    st.error("Batch optimization returned no results.")
                    st.stop()

                # ---- winner row ----
                winner = df_batch_sorted.iloc[0]
                st.subheader("Batch winner (best period + best params)")
                st.json({
                    "period": winner.get("period"),
                    "start": winner.get("start"),
                    "end": winner.get("end"),
                    "objective": winner.get("objective"),
                    "objective_value": float(winner.get("objective_value", np.nan)),
                })

                # ---- build winner spec (params + forced period) ----
                # 1) apply params from row
                winner_spec = build_spec_from_result_row(base_spec, winner)

                # 2) FORCE the period start/end from the batch row (don’t rely on params)
                winner_spec = replace(
                    winner_spec,
                    data=replace(
                        winner_spec.data,
                        start=str(winner.get("start")),
                        end=str(winner.get("end")),
                        include_windows=None,
                        exclude_windows=None,
                    )
                )

                # ---- backtest winner spec ----
                st.divider()
                with st.spinner("Backtesting batch winner..."):
                    winner_bundle = BacktestEngine(winner_spec).run()

                # Persist so the “loaded from session cache” section works
                st.session_state["opt_best_spec"] = winner_spec
                st.session_state["opt_best_bundle"] = winner_bundle
                st.session_state["opt_top_df"] = df_batch_sorted
                st.session_state["opt_best"] = None

                render_bundle(winner_bundle, port_cfg=winner_spec.portfolio)

                # CRITICAL: stop so the global run_optimization below doesn't overwrite the batch winner
                st.stop()

            with st.spinner("Running optimization..."):
                best, top_df, best_params, best_spec, ranked_df = run_optimization(
                    base_spec=base_spec,
                    active_params=active_params,
                    cfg=opt_cfg,
                )

            # ✅ Run best main backtest NOW so we have `best_bundle` available for comparison
            with st.spinner("Backtesting best main strategy..."):
                best_bundle = BacktestEngine(best_spec).run()

            st.session_state["opt_best_bundle"] = best_bundle  # keep your persistence
            # =============================
            # Best-vs-best comparator in OPTIMIZE
            # =============================
            cmp_best_spec = None
            cmp_bundle = None

            if compare_mode == "Another strategy (same data)" and compare_strategy_kind:
                # 1) build comparator base spec (same data/portfolio, comparator strategy)
                # You need baseline params for comparator; simplest: empty dict -> strategy defaults inside engine
                cmp_strategy_params0 = {}  # TODO optionally build baseline UI per strategy kind

                cmp_base_spec = make_base_spec(
                    source_key=source_key,
                    symbol=symbol,
                    timezone=timezone,
                    interval=interval,
                    bmce_tmp_path=tmp_path,
                    start=start_str,
                    end=end_str,
                    include_windows=include_windows,
                    exclude_windows=exclude_windows,
                    yf_period=yf_period,
                    yf_interval=yf_interval,
                    yf_auto_adjust=yf_auto_adjust,
                    strategy_kind=str(compare_strategy_kind),
                    strategy_params=cmp_strategy_params0,
                    allow_short=bool(allow_short),
                    initial_cash=float(initial_cash0),
                    rebalance_policy=str(rebalance_policy0),
                    sizing_mode=str(sizing_mode0),
                    buy_pct_cash=float(buy0),
                    sell_pct_shares=float(sell0),
                    cooldown_bars=int(cooldown0),
                    use_volume_gate=bool(use_volume_gate0),
                    volume_gate_kind=str(volume_gate_kind0),
                    min_volume_abs=float(min_volume_abs0),
                    min_volume_ratio_adv=float(min_volume_ratio_adv0),
                    volume_gate_adv_window=int(volume_gate_adv_window0),
                    use_participation_cap=bool(use_participation_cap0),
                    participation_rate=float(participation_rate0),
                    participation_basis=str(participation_basis0),
                    adv_window=int(adv_window0),
                    cost_model=cost_model,
                    min_return_before_sell=float(min_ret0),
                )

                # 2) define which parameters to optimize for comparator strategy
                cmp_catalog = default_param_catalog(str(compare_strategy_kind))

                # simplest: optimize comparator "default_active" like you did for main
                cmp_default_active = (
                    ["strategy.sma_fast_window", "strategy.sma_slow_window"] if compare_strategy_kind == "ma_cross"
                    else ["strategy.sma_window"] if compare_strategy_kind == "sma_price"
                    else ["strategy.rsi_window", "strategy.rsi_oversold", "strategy.rsi_overbought"] if compare_strategy_kind == "rsi"
                    else ["strategy.macd_fast_window", "strategy.macd_slow_window", "strategy.macd_signal_window"] if compare_strategy_kind == "macd"
                    else ["strategy.bb_window", "strategy.bb_k"] if compare_strategy_kind == "bollinger"
                    else []
                )

                cmp_active_params = []
                for k in cmp_default_active:
                    if k in cmp_catalog:
                        p = cmp_catalog[k]
                        if p.enabled and p.domain is not None and (p.kind != "date_window" or len(p.domain) > 0):
                            cmp_active_params.append(p)

                # 3) optimize comparator
                with st.spinner("Optimizing comparator strategy (best-vs-best)..."):
                    cmp_best, cmp_top_df, cmp_best_params, cmp_best_spec, cmp_ranked_df = run_optimization(
                        base_spec=cmp_base_spec,
                        active_params=cmp_active_params,
                        cfg=opt_cfg,
                    )

                # 4) run backtest on BOTH best specs
                with st.spinner("Backtesting best comparator..."):
                    cmp_bundle = BacktestEngine(cmp_best_spec).run()

                # 5) compare best-vs-best (same analyzer.compare you already wrote)
                analyzer = ResultsAnalyzer(periods_per_year=base_spec.periods_per_year, rf_annual=base_spec.rf_annual)
                comp = analyzer.compare(
                    report_a=best_bundle.report,  # ✅ now defined
                    report_b=cmp_bundle.report,
                    label_a=f"{best_spec.strategy.kind} | {symbol}",
                    label_b=f"{cmp_best_spec.strategy.kind} | {symbol}",
                )

                # inject into display bundle using replace() (frozen-safe)
                rep0 = best_bundle.report
                new_meta = dict(getattr(rep0, "meta", None) or {})
                new_meta["cum_compare_labels"] = {"a": f"{best_spec.strategy.kind} | {symbol}", "b": f"{cmp_best_spec.strategy.kind} | {symbol}"}
                new_plots = dict(getattr(rep0, "plots", None) or {})
                new_plots["cum_vs_bench"] = comp["cum_plot"]
                new_tables = dict(getattr(rep0, "tables", None) or {})
                new_tables["curve_vs_comparator"] = comp["curve_table"]
                best_bundle = replace(best_bundle, report=replace(rep0, meta=new_meta, plots=new_plots, tables=new_tables))


            st.subheader("Best result")
            st.json({
                "pnl": best.pnl,
                "cagr": best.cagr,
                "efficiency": best.efficiency,
                "n_fills": best.n_fills,
                "params": best.params,
                "error": best.error,
            })

            st.subheader("Top candidates")

            # Always hide traded_notional if present
            if "traded_notional" in top_df.columns:
                top_df = top_df.drop(columns=["traded_notional"])

            # Build show_cols AFTER dropping
            core = [c for c in ["signal_label", "signal_date", "cagr", "pnl", "n_fills", "error"] if c in top_df.columns]
            rest = [c for c in top_df.columns if c not in set(core + ["pnl"])]  # optionally hide pnl too
            show_cols = core + rest

            st.dataframe(top_df[show_cols], use_container_width=True)


            if ranked_df is not None and isinstance(ranked_df, pd.DataFrame) and (not ranked_df.empty):
                best5_df = ranked_df.head(5)
                worst5_df = ranked_df.tail(5).sort_values(["pnl","cagr"], ascending=[True, True]).reset_index(drop=True)
                mid_start = max(0, (len(ranked_df) // 2) - 2)
                mid5_df = ranked_df.iloc[mid_start: mid_start + 5].reset_index(drop=True)

                tab_best, tab_best5, tab_mid5, tab_worst5 = st.tabs(["Best", "Best 5", "Mid 5", "Worst 5"])

                def _candidate_selector(df_in: pd.DataFrame, label: str, key_prefix: str):
                    st.dataframe(df_in, use_container_width=True)
                    picked = st.selectbox(
                        f"Select candidate row ({label})",
                        options=list(range(len(df_in))),
                        index=0,
                        key=f"{key_prefix}_pick",
                    )
                    row = df_in.iloc[int(picked)]
                    spec_i = build_spec_from_result_row(base_spec, row)

                    if st.button(f"Run backtest + ledger for {label} #{int(picked)+1}", key=f"{key_prefix}_run"):
                        bundle_a = BacktestEngine(spec_i).run()
                        bundle_b = None
                        if cmp_spec is not None:
                            bundle_b = BacktestEngine(cmp_spec).run()

                        st.dataframe(bundle_a.report.tables.get("trade_ledger", pd.DataFrame()), use_container_width=True)
                        st.dataframe(bundle_a.report.tables.get("trades", pd.DataFrame()), use_container_width=True)
                        st.dataframe(bundle_a.report.tables.get("trade_performance", pd.DataFrame()), use_container_width=True)

                        if bundle_b is not None:
                            st.subheader(f"Compare with {cmp_spec.strategy.kind}")
                            st.dataframe(bundle_b.report.tables.get("trade_ledger", pd.DataFrame()), use_container_width=True)
                            st.dataframe(bundle_b.report.tables.get("trades", pd.DataFrame()), use_container_width=True)
                            st.dataframe(bundle_b.report.tables.get("trade_performance", pd.DataFrame()), use_container_width=True)

                with tab_best:
                    _candidate_selector(ranked_df.head(1).reset_index(drop=True), "Best", "opt_best")
                with tab_best5:
                    _candidate_selector(best5_df, "Best 5", "opt_best5")
                with tab_mid5:
                    _candidate_selector(mid5_df, "Mid 5", "opt_mid5")
                with tab_worst5:
                    _candidate_selector(worst5_df, "Worst 5", "opt_worst5")

            
            st.divider()
            # Auto-run the best configuration backtest (no extra button)
            with st.spinner("Running best backtest..."):
                bundle = BacktestEngine(best_spec).run()
                # ------------------------------------------------------------
                # Attach comparator (Optimize tab)
                # ------------------------------------------------------------
                if compare_mode != "None":
                    label_a = f"{best_spec.strategy.kind} | {symbol}"

                    if compare_mode == "Buy & Hold (same data)":
                        cmp_spec = EngineSpec(
                            data=best_spec.data,
                            indicators=best_spec.indicators,
                            strategy=StrategyConfig(kind="buy_hold", params={"buy_pct_cash": 1.0}),
                            portfolio=best_spec.portfolio,
                            benchmark=BenchmarkConfig(enabled=False),
                            plot_indicators=[],
                            periods_per_year=best_spec.periods_per_year,
                            rf_annual=best_spec.rf_annual,
                        )
                        with st.spinner("Backtesting buy&hold comparator..."):
                            cmp_bundle = BacktestEngine(cmp_spec).run()

                        bundle = attach_comparator_to_bundle(
                            bundle, cmp_bundle,
                            label_a=label_a,
                            label_b=f"buy_hold | {symbol}",
                            periods_per_year=best_spec.periods_per_year,
                            rf_annual=best_spec.rf_annual,
                        )

                    elif compare_mode == "Another strategy (same data)":
                        if (cmp_bundle is not None) and (cmp_best_spec is not None):
                            bundle = attach_comparator_to_bundle(
                                bundle, cmp_bundle,
                                label_a=label_a,
                                label_b=f"{cmp_best_spec.strategy.kind} | {symbol}",
                                periods_per_year=best_spec.periods_per_year,
                                rf_annual=best_spec.rf_annual,
                            )

                # ============================================================
                # Optional: rebuild report with benchmark (no engine changes)
                # ============================================================
                if use_benchmark and bench_source_key and bench_symbol:
                    try:
                        bench_md = load_benchmark_market_data_cached(
                            bench_source_key=bench_source_key,
                            bench_symbol=bench_symbol,
                            timezone=timezone,
                            interval=interval,
                            bmce_path=bench_tmp_path if bench_source_key == "bmce" else None,
                            start=start_str,
                            end=end_str,
                            yf_period=bench_yf_period,
                            yf_interval=bench_yf_interval,
                            yf_auto_adjust=bench_yf_auto_adjust,
                        )

                        analyzer = ResultsAnalyzer(periods_per_year=252, rf_annual=0.0)

                        # robust attribute fetch (adjust once you confirm exact bundle field names)
                        portfolio_result = getattr(bundle, "portfolio_result", None) or getattr(bundle, "portfolio", None) or getattr(bundle, "pres", None)
                        if portfolio_result is None:
                            raise AttributeError("Bundle has no portfolio result attribute (expected portfolio_result/portfolio/pres).")
                        report = analyzer.analyze(
                            portfolio_result=portfolio_result,
                            market_data=bundle.md,
                            symbols=bundle.md.symbols(),
                            features_data=getattr(bundle, "feats", None),
                            plot_indicators=getattr(bundle.report.plots.get("price_panel", {}), "indicator_cols", None) if hasattr(bundle, "report") else None,
                            benchmark_market_data=bench_md,
                            benchmark_symbol=bench_symbol,
                        )

                        # overwrite bundle.report so render_bundle() stays unchanged
                        bundle = replace(bundle, report=report)


                    except Exception as e:
                        st.warning(f"Benchmark could not be applied: {e}")

            # Persist bundle so UI doesn't 'lose' it on rerun
            st.session_state["opt_best_bundle"] = bundle

            # Show the fills ledger (one row per execution) with net_invested
            fills_df = getattr(bundle, "report", None).tables.get("trades", pd.DataFrame()) if getattr(bundle, "report", None) is not None else pd.DataFrame()
            if fills_df is not None and not fills_df.empty and "net_invested" in fills_df.columns:
                st.subheader("Best strategy fills ledger (net_invested)")
                show_cols = [c for c in ["timestamp","symbol","side","qty","price","notional","cost","net_invested","cash_after"] if c in fills_df.columns]
                st.dataframe(fills_df[show_cols], use_container_width=True)

            render_bundle(bundle, port_cfg=best_spec.portfolio,strategy_params=best_spec.strategy.params,)

        finally:
            if bench_is_temp and bench_tmp_path and os.path.exists(bench_tmp_path):
                try:
                    os.remove(bench_tmp_path)
                except OSError:
                    pass
            if bench_tmp_dir and os.path.isdir(bench_tmp_dir):
                try:
                    os.rmdir(bench_tmp_dir)
                except OSError:
                    pass
            if tmp_is_temp and tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            if tmp_dir and os.path.isdir(tmp_dir):
                try:
                    os.rmdir(tmp_dir)
                except OSError:
                    pass
    # ------------------------------------------------------------
    # Persisted optimization display (survives Streamlit reruns)
    # ------------------------------------------------------------
    if (not run_opt) and (st.session_state.get("opt_best_bundle") is not None):
        best = st.session_state.get("opt_best", None)
        best_spec = st.session_state.get("opt_best_spec", None)
        top_df = st.session_state.get("opt_top_df", None)
        bundle = st.session_state.get("opt_best_bundle")

        st.success("Optimization finished (loaded from session cache).")

        if best is not None:
            st.subheader("Best result")
            st.json({
                "pnl": getattr(best, "pnl", None),
                "n_fills": getattr(best, "n_fills", None),
                "params": getattr(best, "params", None),
                "error": getattr(best, "error", None),
            })

        if isinstance(top_df, pd.DataFrame):
            st.subheader("Top candidates")
            show_cols = [c for c in ["pnl","cagr","n_fills","error"] if c in top_df.columns] + \
                        [c for c in top_df.columns if c not in ("pnl","cagr","n_fills","error","traded_notional")]
            # Hide traded_notional from UI if present
            if "traded_notional" in top_df.columns:
                top_df = top_df.drop(columns=["traded_notional"])
            st.dataframe(top_df[show_cols], use_container_width=True)
            fills_df = getattr(bundle, "report", None).tables.get("trades", pd.DataFrame()) if getattr(bundle, "report", None) is not None else pd.DataFrame()
            if fills_df is not None and not fills_df.empty and "net_invested" in fills_df.columns:
                st.subheader("Best strategy fills ledger (net_invested)")
                show_cols = [c for c in ["timestamp","symbol","side","qty","price","notional","cost","net_invested","cash_after"] if c in fills_df.columns]
                st.dataframe(fills_df[show_cols], use_container_width=True)

            render_bundle(bundle, port_cfg=(best_spec.portfolio if best_spec is not None else None))

