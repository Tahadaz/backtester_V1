# results.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExplainItem:
    title: str
    why: str
    latex: Optional[str] = None
    notes: Optional[str] = None
    columns: Optional[Dict[str, str]] = None   # for tables: col -> meaning


@dataclass(frozen=True)
class BacktestReport:
    """
    Canonical, UI-agnostic backtest report container.

    NOTE: This class is consumed directly by the Streamlit app.  The `explain`
    field is critical for the in-app popovers / tooltips – do not remove it.
    """
    metrics: Dict[str, float]
    series: Dict[str, pd.Series]
    tables: Dict[str, pd.DataFrame]
    plots: Dict[str, Any]
    style: Dict[str, Any]
    explain: Dict[str, ExplainItem] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


class ResultsAnalyzer:
    """
    Compute-only reporting layer.

    Responsibilities:
      - Compute strategy performance series/metrics
      - Build "summary tables" (Curve vs Benchmark / Trade / Time)
      - Build month heatmap matrix and yearly returns
      - Prepare plot-ready payloads (Series/DataFrames) for the UI

    Non-responsibilities:
      - No matplotlib/plotly/streamlit rendering
      - No colormaps, fills, marker styles (UI layer handles that)
    """

    def __init__(self, periods_per_year: int = 252, rf_annual: float = 0.0) -> None:
        self.periods_per_year = int(periods_per_year)
        self.rf_annual = float(rf_annual)

    # =============================
    # Public API
    # =============================
    def analyze(
        self,
        portfolio_result,
        market_data,
        symbols: Sequence[str],
        features_data=None,
        plot_indicators: Optional[List[str]] = None,
        benchmark_market_data=None,
        benchmark_symbol: Optional[str] = None,
    ) -> BacktestReport:
        symbols = list(symbols)
        if not symbols:
            raise ValueError("symbols must be non-empty")

        # --- core strategy series ---
        equity = portfolio_result.equity_curve.astype(float).sort_index()
        rets = portfolio_result.returns.astype(float).reindex(equity.index).fillna(0.0)
        explain: Dict[str, ExplainItem] = {
            # --- metrics ---
            "metric.pnl": ExplainItem(
                title="PnL (currency)",
                why="Profit in account currency. Primary optimizer objective.",
                latex=r"\mathrm{PnL}=Equity_T-Equity_0",
                notes="Equity includes cash + marked-to-market positions."
            ),
            "metric.efficiency": ExplainItem(
                title="Efficiency (PnL / volume invested)",
                why=("Normalizes PnL by net cash invested (buys - sells). "
                     "The invested volume is reset after the last SELL of each calendar month."),
                latex=(
                    r"\mathrm{VolumeInv}=\sum_{m}\left(\sum_{k\in m} \mathrm{signed\_notional}_k\right)\;\text{(reset at last SELL of month)}"
                    "\n"
                    r"\mathrm{Eff}=\begin{cases}1,&\mathrm{VolumeInv}\le 0\\ \frac{\mathrm{PnL}}{\mathrm{VolumeInv}},&\mathrm{VolumeInv}>0\end{cases}"
                ),
                notes="signed_notional = +notional for BUY, -notional for SELL. If VolumeInv <= 0 → set efficiency to 100% (1.0)."
            ),
            "metric.total_return": ExplainItem(
                title="Total return",
                why="Total compounded return over the period.",
                latex=r"R_{tot}=\prod_{t=1}^{T}(1+r_t)-1",
            ),
            "metric.cagr": ExplainItem(
                title="CAGR",
                why="Annualized compounded return (comparable across backtests).",
                latex=r"\mathrm{CAGR}=\left(\prod_{t=1}^{T}(1+r_t)\right)^{\frac{ppY}{T}}-1",
                notes="ppY = periods_per_year (e.g. 252)."
            ),

            # --- tables ---
            "table.trade_ledger": ExplainItem(
                title="Trade ledger (FIFO round-trips)",
                why="Closed trades built by matching fills using FIFO lots.",
                columns={
                    "entry_time": "Entry timestamp",
                    "exit_time": "Exit timestamp",
                    "side": "LONG/SHORT",
                    "qty": "Closed quantity",
                    "gross_pnl": "PnL before costs",
                    "net_pnl": "PnL after entry+exit costs",
                    "return_pct": "net_pnl / (qty * entry_price)",
                    "hold_days": "Holding period in days",
                },
            ),

            # --- plots ---
            "plot.price_panel": ExplainItem(
                title="Price + indicator + trades",
                why="Shows price series + strategy indicator and where trades occurred.",
                notes="Trades are placed at open(t+1) and MTM at close(t+1) per your portfolio convention."
            ),
            "plot.cum_vs_bench": ExplainItem(
                title="Cumulative returns vs benchmark",
                why="Compares strategy performance to benchmark (if provided).",
                latex=r"CR_t=\prod_{u\le t}(1+r_u)-1"
            ),
        }


        cum = (1.0 + rets).cumprod() - 1.0
        dd = self._drawdown_from_equity(equity)
        pnl = equity.diff().fillna(0.0)              # currency PnL per bar
        cum_pnl = pnl.cumsum()


        # --- price + indicators payload (single asset for now) ---
        sym0 = symbols[0]
        px = market_data.bars[sym0]["Close"].reindex(equity.index).astype(float)
        bars0 = market_data.bars[sym0].reindex(equity.index).copy()

        # ensure float for plot libs
        for c in ["Open", "High", "Low", "Close"]:
            if c in bars0.columns:
                bars0[c] = bars0[c].astype(float)

        ind_df = None
        if features_data is not None and plot_indicators:
            feats_sym = features_data.features[sym0]
            cols = [c for c in plot_indicators if c in feats_sym.columns]
            if cols:
                ind_df = feats_sym[cols].reindex(bars0.index)
        
        # trades payload (fills)
        init_cash = float(portfolio_result.meta.get("config", {}).get("initial_cash", 0.0))
        sym0 = symbols[0]
        # -------------------------
        # Fills: keep RAW + DISPLAY
        # -------------------------
        raw_fills = portfolio_result.trades  # canonical: timestamp,symbol,qty,price,cost,...

        trades = self._prepare_trades_table(
            raw_fills,
            initial_cash=init_cash,
            mark_price=px,
        )

        # Compute artifacts MUST use raw_fills (not the prettified table)
        trade_ledger = self._trade_ledger_from_fills(raw_fills)
        trade_perf   = self._trade_performance_summary(trade_ledger)

        volume_inv   = self._volume_invested_reset_last_sell_monthly(raw_fills)
        round_trips  = self._round_trips_from_fills(raw_fills)

        pnl_total = float(equity.iloc[-1] - equity.iloc[0]) if len(equity) else 0.0
        efficiency = 1.0 if volume_inv <= 0 else float(pnl_total / volume_inv)
        # --- benchmark series (optional) ---
        bench_rets = None
        bench_cum = None
        bench_equity = None
        bench_dd = None
        rel_cum = None

        if benchmark_market_data is not None:
            bsym = benchmark_symbol or list(benchmark_market_data.bars.keys())[0]
            bpx = benchmark_market_data.bars[bsym]["Close"].astype(float)
            bpx = bpx.reindex(equity.index).ffill()

            bench_rets = bpx.pct_change().fillna(0.0)
            bench_equity = (1.0 + bench_rets).cumprod()
            bench_cum = bench_equity - 1.0
            bench_dd = self._drawdown_from_equity(bench_equity)
            rel_cum = (1.0 + rets).cumprod() / (1.0 + bench_rets).cumprod() - 1.0

        # --- monthly/yearly returns ---
        monthly_mat = self._monthly_return_matrix(rets)   # year x month (1..12)
        yearly = self._yearly_returns(rets)               # index=year, values=return

        # --- round trips from fills (needed for Trade table) ---
        round_trips = self._round_trips_from_fills(trades)

        # --- 3 summary tables like your screenshot ---
        curve_vs_bench = self._curve_vs_benchmark_table(
            strat_rets=rets,
            strat_equity=(1.0 + rets).cumprod(),
            strat_dd=dd,
            round_trips=round_trips,
            bench_rets=bench_rets,
            bench_equity=bench_equity,
            bench_dd=bench_dd,
        )
        trade_tbl = self._trade_table(round_trips)
        time_tbl = self._time_table(rets)

        # --- headline metrics (for quick display) ---
        metrics = self._headline_metrics(rets, dd, bench_rets)
        metrics["Net PnL"] = float(pnl_total)
        metrics["VolumeInv"] = float(volume_inv)
        metrics["Efficiency"] = float(efficiency)

        # --- time series table ---
        ts = pd.DataFrame(
            {
                "equity": equity,
                "returns": rets,
                "cum_returns": cum,
                "drawdown": dd,
            },
            index=equity.index,
        )
        if bench_cum is not None:
            ts["bench_cum_returns"] = bench_cum
            ts["rel_cum_returns"] = rel_cum

        # --- outputs ---
        tables: Dict[str, pd.DataFrame] = {
            "trades": trades,
            "timeseries": ts,
            "curve_vs_benchmark": curve_vs_bench,
            "trade_summary": trade_tbl,
            "time_summary": time_tbl,
            "monthly_returns": monthly_mat,
            "yearly_returns": yearly.to_frame("year_return"),
            "trade_ledger":trade_ledger,
            "trade_performance":trade_perf,
        }

        plots: Dict[str, Any] = {
            "price_panel": {
                "symbol": sym0,
                "bars": bars0,            # <-- NEW: full OHLC
                "price": bars0["Close"],  # optional convenience
                "indicators": ind_df,
                "trades": trades,
                "indicator_cols": plot_indicators,
            },
            "cum_vs_bench": {"strategy": cum, "benchmark": bench_cum},
            "drawdown": dd,
            "monthly_heatmap": monthly_mat,  # app will render + annotate
            "yearly_bar": yearly,
        }

        series: Dict[str, pd.Series] = {
            "equity": equity,
            "returns": rets,
            "cum_returns": cum,
            "drawdown": dd,
            "pnl":pnl,
            "cum_pnl":cum_pnl,
        }
        if bench_cum is not None:
            series.update(
                {
                    "bench_returns": bench_rets,
                    "bench_cum_returns": bench_cum,
                    "rel_cum_returns": rel_cum,
                }
            )

        style = self._style_spec()

        try:
            rep = BacktestReport(
                metrics=metrics,
                series=series,
                tables=tables,
                plots=plots,
                style=style,
                explain=explain,
                meta={
                    "symbols": symbols,
                    "benchmark_symbol": benchmark_symbol,
                },
            )
            return rep
        except Exception as e:
            # Defensive logging – helps debugging if construction ever fails
            print("FAILED building BacktestReport:", type(e), e)
            print(
                "types:",
                type(metrics),
                type(series),
                type(tables),
                type(plots),
                type(style),
            )
            raise

    def compare(
    self,
    report_a: BacktestReport,
    report_b: BacktestReport,
    label_a: str = "Strategy",
    label_b: str = "Benchmark",
) -> Dict[str, Any]:
        """
        Build comparison artifacts (cum returns plot payload + curve-vs table)
        from two BacktestReports.
        """

        ts_a = report_a.tables["timeseries"].copy()
        ts_b = report_b.tables["timeseries"].copy()

        # Align on common dates (intersection = robust)
        idx = ts_a.index.intersection(ts_b.index).sort_values()
        ts_a = ts_a.reindex(idx)
        ts_b = ts_b.reindex(idx)

        # Robust numeric returns
        rets_a = pd.to_numeric(ts_a["returns"], errors="coerce").fillna(0.0).astype(float)
        rets_b = pd.to_numeric(ts_b["returns"], errors="coerce").fillna(0.0).astype(float)

        eq_a = (1.0 + rets_a).cumprod()
        eq_b = (1.0 + rets_b).cumprod()

        dd_a = self._drawdown_from_equity(eq_a)
        dd_b = self._drawdown_from_equity(eq_b)

        cum_a = eq_a - 1.0
        cum_b = eq_b - 1.0

        # Round trips from fills (if any)
        rt_a = self._round_trips_from_fills(report_a.tables.get("trades", pd.DataFrame()))
        rt_b = self._round_trips_from_fills(report_b.tables.get("trades", pd.DataFrame()))

        # Build curve-vs table using existing machinery
        table = self._curve_vs_benchmark_table(
            strat_rets=rets_a,
            strat_equity=eq_a,
            strat_dd=dd_a,
            round_trips=rt_a,
            bench_rets=rets_b,
            bench_equity=eq_b,
            bench_dd=dd_b,
        )

        # Rename columns from Strategy/Benchmark -> actual labels
        cols = [str(c) for c in table.columns]
        if len(cols) == 1 and cols[0].strip().lower() == "strategy":
            table.columns = [label_a]
        elif len(cols) >= 2 and cols[0].strip().lower() == "strategy" and cols[1].strip().lower() == "benchmark":
            table.columns = [label_a, label_b] + cols[2:]  # keep any extra cols if you add later

        return {
            "cum_plot": {"strategy": cum_a, "benchmark": cum_b},
            "curve_table": table,
            "cum_df": pd.DataFrame({label_a: cum_a, label_b: cum_b}, index=idx),
        }



    # =============================
    # Styling hints for the UI
    # =============================
    def _style_spec(self) -> Dict[str, Any]:
        return {
            # metrics where higher is better
            "good_high": [
                "Total Return",
                "CAGR",
                "Sharpe Ratio",
                "Sortino Ratio",
                "R-Squared",
                "Trade Winning %",
                "Average Trade %",
                "Average Win %",
                "Best Trade %",
                "Winning Months %",
                "Average Winning Month %",
                "Best Month %",
                "Winning Years %",
                "Best Year %",
            ],
            # metrics where lower is better (or "less negative" drawdown)
            "good_low": [
                "Annual Volatility",
                "Max Daily Drawdown",
                "Max Drawdown Duration",
                "Average Losing Month %",
                "Worst Month %",
                "Worst Year %",
                "Average Loss %",
                "Worst Trade %",
            ],
        }

    # =============================
    # Core series metrics
    # =============================
    def _drawdown_from_equity(self, equity: pd.Series) -> pd.Series:
        e = equity.astype(float).copy()
        peak = e.cummax()
        dd = e / peak - 1.0
        return dd.fillna(0.0)

    def _annualized_return(self, rets: pd.Series) -> float:
        r = rets.dropna()
        n = len(r)
        if n <= 1:
            return np.nan
        total = (1.0 + r).prod()
        return float(total ** (self.periods_per_year / n) - 1.0)

    def _annualized_vol(self, rets: pd.Series) -> float:
        r = rets.dropna()
        if len(r) <= 1:
            return np.nan
        return float(r.std(ddof=1) * np.sqrt(self.periods_per_year))

    def _sharpe(self, rets: pd.Series) -> float:
        vol = self._annualized_vol(rets)
        if vol == 0 or np.isnan(vol):
            return np.nan
        return float((self._annualized_return(rets) - self.rf_annual) / vol)

    def _sortino(self, rets: pd.Series) -> float:
        r = rets.dropna()
        if r.empty:
            return np.nan
        downside = r[r < 0]
        if downside.empty:
            return np.inf
        downside_dev = downside.std(ddof=1) * np.sqrt(self.periods_per_year)
        if downside_dev == 0 or np.isnan(downside_dev):
            return np.nan
        return float((self._annualized_return(r) - self.rf_annual) / downside_dev)

    def _headline_metrics(self, rets: pd.Series, dd: pd.Series, bench_rets: Optional[pd.Series]) -> Dict[str, float]:
        out = {
            "Total return": float((1.0 + rets).prod() - 1.0),
            "CAGR": float(self._annualized_return(rets)),
            "Vol (ann.)": float(self._annualized_vol(rets)),
            "Sharpe": float(self._sharpe(rets)),
            "Sortino": float(self._sortino(rets)),
            "Max drawdown": float(dd.min()) if dd is not None and len(dd) else np.nan,
        }
        if bench_rets is not None:
            out["Bench total return"] = float((1.0 + bench_rets).prod() - 1.0)
        return out

    # =============================
    # Trades / fills
    # =============================
    def _prepare_trades_table(
        self,
        trades: pd.DataFrame,
        initial_cash: float,
        *,
        mark_price: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Fills ledger + CMP (weighted average cost) + realized/latent PnL.

        Keeps your existing audit fields:
        timestamp, symbol, qty, side, price, notional, cost,
        signed_notional, net_invested, cash_after,
        month_net_invested, is_last_sell_in_month, month_volumeinv_at_last_sell, month

        Adds requested columns:
        trade_date, quantity, fees,
        available_quantity, cmp,
        position_value_cost (= cmp * available_quantity),
        pnl_realise, pnl_latent

        Important:
        - Fees are capitalized into CMP when opening/increasing a position.
        - Fees are deducted from realized PnL when closing/reducing (pro-rated on flips).
        - Latent PnL marked to `mark_price` at trade timestamp when provided, else uses execution price.
        """


        if trades is None or trades.empty:
            out_cols = [
                "side", "price", "fees",
                "available_quantity", "cmp", "position_value_cost",
                "pnl_realise", "pnl_latent",
            ]   
            # remove dupes while preserving order
            seen = set()
            out_cols = [c for c in out_cols if not (c in seen or seen.add(c))]
            return pd.DataFrame(columns=out_cols)

        df = trades.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)

        # numeric
        for c in ["qty", "price", "notional", "cost", "commission_ht", "vat", "slippage"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

        # side
        if "side" not in df.columns:
            df["side"] = np.where(df["qty"].astype(float) > 0, "BUY", "SELL")
        df["side"] = df["side"].astype(str).str.upper()

        # cost
                # cost (ensure exists)
        if "cost" not in df.columns:
            df["cost"] = 0.0
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0).astype(float)


        # notional (absolute)   
        if "notional" not in df.columns:
            df["notional"] = (df["qty"].abs() * df["price"]).astype(float)

        # signed notional (your convention)
        df["signed_notional"] = np.where(df["qty"].astype(float) >= 0, df["notional"], -df["notional"]).astype(float)

        keep = [c for c in [
            "timestamp", "symbol", "qty", "side", "price", "notional", "cost",
        ] if c in df.columns]
        df = df[keep].sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        # account-level running net invested + cash path (same as your current)

        # requested aliases
        df["fees"] = df["cost"].astype(float)

        # mark price
        if mark_price is not None and isinstance(mark_price, pd.Series) and not mark_price.empty:
            mp = mark_price.copy()
            mp.index = pd.to_datetime(mp.index)
            # map exact timestamps; if your timestamps are daily, this matches
            df["_mark"] = df["timestamp"].map(mp.to_dict())
            df["_mark"] = pd.to_numeric(df["_mark"], errors="coerce")
        else:
            df["_mark"] = np.nan

        # CMP accounting columns
        df["available_quantity"] = np.nan
        df["cmp"] = np.nan
        df["position_value_cost"] = np.nan   # NEW NAME: cmp * available_quantity
        df["pnl_realise"] = 0.0
        df["pnl_latent"] = np.nan

        for sym, g_idx in df.groupby("symbol").groups.items():
            pos = 0.0
            cmp_ = 0.0

            def fee_split(total_fee: float, close_abs: float, total_abs: float) -> tuple[float, float]:
                if total_abs <= 0:
                    return 0.0, total_fee
                close_fee = total_fee * (close_abs / total_abs)
                return close_fee, total_fee - close_fee

            for i in g_idx:
                qty_signed = float(df.at[i, "qty"])
                q = abs(qty_signed)
                px = float(df.at[i, "price"])
                fee = float(df.at[i, "fees"])
                mark = df.at[i, "_mark"]
                if not np.isfinite(mark):
                    mark = px

                prev_pos = pos
                prev_cmp = cmp_

                realised = 0.0

                # Open/add same direction
                if prev_pos == 0 or np.sign(prev_pos) == np.sign(qty_signed):
                    new_pos = prev_pos + qty_signed
                    old_basis = abs(prev_pos) * prev_cmp
                    add_basis = q * px + fee
                    new_basis = old_basis + add_basis
                    cmp_ = (new_basis / abs(new_pos)) if abs(new_pos) > 0 else 0.0
                    pos = new_pos

                # Reduce/close/flip
                else:
                    close_abs = min(abs(prev_pos), q)
                    open_abs = max(0.0, q - close_abs)
                    close_fee, open_fee = fee_split(fee, close_abs, q)

                    if prev_pos > 0:
                        # selling long
                        realised = close_abs * (px - prev_cmp) - close_fee
                    else:
                        # buying back short
                        realised = close_abs * (prev_cmp - px) - close_fee

                    pos = prev_pos + qty_signed

                    if pos == 0:
                        cmp_ = 0.0
                    else:
                        # flip => remainder opens new position with open_fee in basis
                        if np.sign(pos) != np.sign(prev_pos):
                            cmp_ = (open_abs * px + open_fee) / open_abs if open_abs > 0 else px
                        else:
                            # partial close => CMP unchanged
                            cmp_ = prev_cmp

                df.at[i, "pnl_realise"] = float(realised)
                df.at[i, "available_quantity"] = float(pos)
                df.at[i, "cmp"] = float(cmp_)
                df.at[i, "position_value_cost"] = float(cmp_ * pos)

                if pos == 0:
                    df.at[i, "pnl_latent"] = 0.0
                elif pos > 0:
                    df.at[i, "pnl_latent"] = float(pos * (mark - cmp_))
                else:
                    df.at[i, "pnl_latent"] = float(abs(pos) * (cmp_ - mark))

        

        # final order: requested columns first (plus keep fills columns)

        # ============================================================
        # FINAL PRESENTATION: rename, de-duplicate, and reorder columns
        # ============================================================

        # 1) Rename columns for display (French labels)
        #    NOTE: pandas supports any string column names; just be consistent downstream.
        rename_map = {
            "price": "prix d'éxécution (open du jour)",
            "_mark": "close du jour",
            "cmp": "cmp",
            "pnl_realise": "pnl realisé",
            "pnl_latent": "pnl latent",
            "qty": "quantité",
        }
        for k in list(rename_map.keys()):
            if k not in df.columns:
                rename_map.pop(k, None)
        df = df.rename(columns=rename_map)

        # 2) Remove redundant columns
        # You currently have both:
        # - "cost" (total fees you computed in engine)
        # - "fees" (alias of cost)
        # Keep ONE. Here: keep "cost" and drop "fees".
        drop_if_exists = [
            "fees",          # duplicated with cost
            "trade_date",    # duplicated with timestamp (unless you want both)
            "quantity",      # duplicated with qty.abs() (unless you prefer quantity over qty)
            "notional",
            "commission_ht", # drop components
            "vat",
            "slippage",      # you said you don't have it, but harmless if present
            "trade_date",    # duplicate of timestamp
        ]
        drop_if_exists = [c for c in drop_if_exists if c in df.columns]
        df = df.drop(columns=drop_if_exists)

        # Optional: if you prefer "quantity" instead of signed "qty"
        # uncomment:
        # df["quantity"] = df["qty"].abs().astype(float)
        # df = df.drop(columns=["qty"])
        # and then include "quantity" in the display order below.

        # 3) Build a “front” layout (human-friendly)
        front = [
            "timestamp",
            "symbol",
            "side",
            "prix d'éxécution (open du jour)",
            "quantité",  # signed quantity (keep); or use "quantity" if you switch as noted above
            "cmp",
            "pnl realisé",
            "pnl latent",
            "close du jour",
            "available_quantity",
            "position_value_cost",
            "cost",  # total fees
        ]


        # 4) Keep the rest (audit fields) after, without duplicates
        #    Also preserve any columns that exist but aren’t listed in `front`.
        existing_front = [c for c in front if c in df.columns]
        audit_preferred = [
                    "notional",
                    "signed_notional",
                    "net_invested",
                    "cash_after",
                ]

        audit_preferred = [c for c in audit_preferred if c in df.columns and c not in existing_front]

        used = set(existing_front + audit_preferred)
        rest = [c for c in df.columns if c not in used]

        df = df[existing_front + audit_preferred + rest]
        return df



    
    def _volume_invested_reset_last_sell_monthly(self, fills: pd.DataFrame) -> float:
        """Compute VolumeInv with a monthly reset at the *last* SELL of each month.

        Definition requested:
          VolumeInv = (cash bought with) - (cash sold with)
                    = Σ signed_notional, where signed_notional = +notional for BUY, -notional for SELL.

        Reset rule:
          For each calendar month, we treat the last SELL fill in that month as a reset point.
          Operationally, we compute the net invested *per month* (using fills up to and including
          that last SELL when it exists), then sum across months.

        Notes:
          - If a month has no SELL fills, we use the whole month's Σ signed_notional.
          - This is a backtest-level scalar (used as the denominator for Efficiency).
        """
        if fills is None or fills.empty:
            return 0.0

        f = fills.copy()
        if "timestamp" not in f.columns:
            return 0.0

        f["timestamp"] = pd.to_datetime(f["timestamp"], errors="coerce")
        f = f.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # Prefer signed_notional if present (comes from _prepare_trades_table).
        if "signed_notional" in f.columns:
            sn = pd.to_numeric(f["signed_notional"], errors="coerce").fillna(0.0).astype(float)
        else:
            # Fallback: infer from qty/price/notional if needed.
            if "notional" in f.columns:
                notional = pd.to_numeric(f["notional"], errors="coerce").fillna(0.0).astype(float)
            elif ("qty" in f.columns) and ("price" in f.columns):
                q = pd.to_numeric(f["qty"], errors="coerce").fillna(0.0).astype(float)
                p = pd.to_numeric(f["price"], errors="coerce").fillna(0.0).astype(float)
                notional = (q.abs() * p).astype(float)
            else:
                return 0.0

            # Determine side: prefer explicit 'side', else sign of qty.
            if "side" in f.columns:
                side = f["side"].astype(str).str.upper()
                sn = np.where(side == "SELL", -notional, notional).astype(float)
                sn = pd.Series(sn, index=f.index)
            elif "qty" in f.columns:
                q = pd.to_numeric(f["qty"], errors="coerce").fillna(0.0).astype(float)
                sn = np.where(q < 0, -notional, notional).astype(float)
                sn = pd.Series(sn, index=f.index)
            else:
                return 0.0

        f["_signed_notional"] = sn

        # SELL mask
        if "side" in f.columns:
            sell_mask = f["side"].astype(str).str.upper().eq("SELL")
        elif "qty" in f.columns:
            sell_mask = pd.to_numeric(f["qty"], errors="coerce").fillna(0.0).astype(float) < 0
        else:
            sell_mask = pd.Series(False, index=f.index)

        f["_month"] = f["timestamp"].dt.to_period("M")

        vol_total = 0.0
        for m, g in f.groupby("_month", sort=True):
            g = g.sort_values("timestamp")
            g_sells = g[sell_mask.loc[g.index]]
            if not g_sells.empty:
                last_sell_ts = g_sells["timestamp"].max()
                g_use = g[g["timestamp"] <= last_sell_ts]
            else:
                g_use = g

            vol_total += float(g_use["_signed_notional"].sum())

        return float(vol_total)
    def _round_trips_from_fills(self, fills: pd.DataFrame) -> pd.DataFrame:
        """
        Build round-trip trades from fills.
        Assumes fills change position over time; closes when position goes to 0 or flips sign.
        """
        if fills is None or fills.empty:
            return pd.DataFrame(columns=[
                "symbol","entry_ts","exit_ts","side","entry_price","exit_price","qty",
                "pnl","ret","days"
            ])

        df = fills.copy()

        if "qty" not in df.columns and "quantité" in df.columns:
            df["qty"] = pd.to_numeric(df["quantité"], errors="coerce")
        if "price" not in df.columns and "prix d'éxécution (open du jour)" in df.columns:
            df["price"] = pd.to_numeric(df["prix d'éxécution (open du jour)"], errors="coerce")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        out_rows = []
        for sym, g in df.groupby("symbol", sort=False):
            pos = 0.0
            entry_ts = None
            entry_price = None
            entry_qty = 0.0
            entry_side = None
            entry_notional = None
            entry_cost = 0.0

            for _, r in g.iterrows():
                ts = r["timestamp"]
                qty = float(r["qty"])
                px = float(r["price"])
                cost = float(r["cost"]) if "cost" in r and pd.notna(r["cost"]) else 0.0

                prev_pos = pos
                pos = prev_pos + qty

                # open when 0 -> nonzero
                if prev_pos == 0 and pos != 0:
                    entry_ts = ts
                    entry_price = px
                    entry_qty = pos
                    entry_side = "LONG" if pos > 0 else "SHORT"
                    entry_notional = abs(entry_qty * entry_price)
                    entry_cost = cost
                    continue

                # close when back to 0 OR flip sign
                closing = (prev_pos != 0 and pos == 0) or (prev_pos != 0 and pos != 0 and np.sign(prev_pos) != np.sign(pos))
                if closing and entry_ts is not None:
                    exit_ts = ts
                    exit_price = px
                    closed_qty = abs(prev_pos)  # quantity actually closed at this fill

                    if prev_pos > 0:
                        pnl = (exit_price - entry_price) * closed_qty
                    else:
                        pnl = (entry_price - exit_price) * closed_qty

                    # net: subtract entry_cost + this fill cost (still imperfect if entry_cost was for bigger size)
                    pnl_net = pnl - entry_cost - cost

                    notional_entry = abs(entry_price * closed_qty)
                    ret = pnl_net / notional_entry if notional_entry > 0 else np.nan

                    days = int((exit_ts - entry_ts).days)

                    out_rows.append({
                        "symbol": sym,
                        "entry_ts": entry_ts,
                        "exit_ts": exit_ts,
                        "side": entry_side,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "qty": closed_qty,
                        "net_pnl": pnl_net,
                        "ret": ret,
                        "days": days,
                    })

                    # if flip sign, start new trade immediately at same fill
                    if pos != 0:
                        entry_ts = ts
                        entry_price = px
                        entry_qty = pos
                        entry_side = "LONG" if pos > 0 else "SHORT"
                        entry_notional = abs(entry_qty * entry_price)
                        entry_cost = cost
                    else:
                        entry_ts = None
                        entry_price = None
                        entry_qty = 0.0
                        entry_side = None
                        entry_notional = None
                        entry_cost = 0.0

        return pd.DataFrame(out_rows)

    def _trades_per_year(self, idx: pd.DatetimeIndex, round_trips: pd.DataFrame) -> float:
        if idx is None or len(idx) < 2:
            return np.nan
        years = (idx[-1] - idx[0]).days / 365.25
        if years <= 0:
            return np.nan
        return float((0 if round_trips is None else len(round_trips)) / years)

    # =============================
    # Monthly / yearly returns
    # =============================
    def _monthly_return_matrix(self, rets: pd.Series) -> pd.DataFrame:
        r = rets.copy()
        r.index = pd.to_datetime(r.index)
        m = r.resample("M").apply(lambda x: (1.0 + x).prod() - 1.0)
        if m.empty:
            return pd.DataFrame()
        df = m.to_frame("ret")
        df["year"] = df.index.year
        df["month"] = df.index.month
        return df.pivot(index="year", columns="month", values="ret").sort_index()

    def _yearly_returns(self, rets: pd.Series) -> pd.Series:
        r = rets.copy()
        r.index = pd.to_datetime(r.index)
        y = r.resample("YE").apply(lambda x: (1.0 + x).prod() - 1.0)
        if y.empty:
            return pd.Series(dtype=float)
        y.index = y.index.year
        y.name = "year_return"
        return y

    # =============================
    # Summary tables (3 panels)
    # =============================
    def _max_dd_duration(self, equity: pd.Series) -> int:
        eq = equity.dropna()
        if eq.empty:
            return 0
        peak = eq.cummax()
        underwater = eq < peak
        max_run, run = 0, 0
        for u in underwater.values:
            if bool(u):
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0
        return int(max_run)

    def _r_squared(self, strat_rets: pd.Series, bench_rets: pd.Series) -> float:
        s, b = strat_rets.align(bench_rets, join="inner")
        s = s.fillna(0.0)
        b = b.fillna(0.0)
        if len(s) < 3:
            return np.nan
        corr = np.corrcoef(s.values, b.values)[0, 1]
        return float(corr * corr) if not np.isnan(corr) else np.nan

    def _curve_vs_benchmark_table(
        self,
        strat_rets: pd.Series,
        strat_equity: pd.Series,
        strat_dd: pd.Series,
        round_trips: pd.DataFrame,
        bench_rets: Optional[pd.Series],
        bench_equity: Optional[pd.Series],
        bench_dd: Optional[pd.Series],
    ) -> pd.DataFrame:
        def pack(rets, equity, dd):
            total = (1.0 + rets).prod() - 1.0
            cagr = self._annualized_return(rets)
            sharpe = self._sharpe(rets)
            sortino = self._sortino(rets)
            vol = self._annualized_vol(rets)
            maxdd = float(dd.min()) if dd is not None and len(dd) else np.nan
            dd_dur = self._max_dd_duration(equity)
            return float(total), float(cagr), float(sharpe), float(sortino), float(vol), float(maxdd), float(dd_dur)

        s_total, s_cagr, s_sh, s_so, s_vol, s_mdd, s_dddur = pack(strat_rets, strat_equity, strat_dd)
        tpy = self._trades_per_year(strat_equity.index, round_trips)

        index = [
            "Total Return",
            "CAGR",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Annual Volatility",
            "R-Squared",
            "Max Daily Drawdown",
            "Max Drawdown Duration",
            "Trades per Year",
        ]

        if bench_rets is not None and bench_equity is not None and bench_dd is not None:
            b_total, b_cagr, b_sh, b_so, b_vol, b_mdd, b_dddur = pack(bench_rets, bench_equity, bench_dd)
            r2 = self._r_squared(strat_rets, bench_rets)

            df = pd.DataFrame(
                {
                    "Strategy": [s_total, s_cagr, s_sh, s_so, s_vol, r2, s_mdd, s_dddur, tpy],
                    "Benchmark": [b_total, b_cagr, b_sh, b_so, b_vol, np.nan, b_mdd, b_dddur, np.nan],
                },
                index=index,
            )
            return df

        df = pd.DataFrame({"Strategy": [s_total, s_cagr, s_sh, s_so, s_vol, np.nan, s_mdd, s_dddur, tpy]}, index=index)
        return df

    def _trade_table(self, round_trips: pd.DataFrame) -> pd.DataFrame:
        idx = [
            "Trade Winning %",
            "Average Trade %",
            "Average Win %",
            "Average Loss %",
            "Best Trade %",
            "Worst Trade %",
            "Worst Trade Date",
            "Avg Days in Trade",
            "Trades",
        ]
        if round_trips is None or round_trips.empty:
            return pd.DataFrame({"Value": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "TBD", np.nan, 0]}, index=idx)

        r = round_trips["ret"].astype(float)
        wins = r[r > 0]
        losses = r[r < 0]

        win_pct = (len(wins) / len(r)) * 100.0 if len(r) else np.nan
        avg_trade = r.mean() * 100.0
        avg_win = wins.mean() * 100.0 if len(wins) else np.nan
        avg_loss = losses.mean() * 100.0 if len(losses) else np.nan
        best = r.max() * 100.0
        worst = r.min() * 100.0

        worst_date = "TBD"
        if len(r):
            worst_row = round_trips.loc[r.idxmin()]
            worst_date = pd.to_datetime(worst_row["exit_ts"]).date().isoformat()

        avg_days = float(round_trips["days"].mean()) if "days" in round_trips.columns else np.nan
        ntr = int(len(round_trips))

        return pd.DataFrame({"Value": [win_pct, avg_trade, avg_win, avg_loss, best, worst, worst_date, avg_days, ntr]}, index=idx)

    def _time_table(self, rets: pd.Series) -> pd.DataFrame:
        idx = [
            "Winning Months %",
            "Average Winning Month %",
            "Average Losing Month %",
            "Best Month %",
            "Worst Month %",
            "Winning Years %",
            "Best Year %",
            "Worst Year %",
        ]

        r = rets.copy()
        r.index = pd.to_datetime(r.index)

        m = r.resample("M").apply(lambda x: (1.0 + x).prod() - 1.0)
        y = r.resample("YE").apply(lambda x: (1.0 + x).prod() - 1.0)

        def stats(x: pd.Series) -> Tuple[float, float, float, float, float]:
            x = x.dropna()
            if x.empty:
                return (np.nan, np.nan, np.nan, np.nan, np.nan)
            wins = x[x > 0]
            losses = x[x < 0]
            win_pct = (len(wins) / len(x)) * 100.0 if len(x) else np.nan
            avg_win = wins.mean() * 100.0 if len(wins) else np.nan
            avg_loss = losses.mean() * 100.0 if len(losses) else np.nan
            best = x.max() * 100.0
            worst = x.min() * 100.0
            return (win_pct, avg_win, avg_loss, best, worst)

        m_win, m_avgw, m_avgl, m_best, m_worst = stats(m)
        y_win, _, _, y_best, y_worst = stats(y)

        return pd.DataFrame({"Value": [m_win, m_avgw, m_avgl, m_best, m_worst, y_win, y_best, y_worst]}, index=idx)
    def _trade_ledger_from_fills(self, fills: pd.DataFrame) -> pd.DataFrame:
        """
        Build closed-trade ledger (round trips) from fills using FIFO lot matching.

        Expected fills columns:
        timestamp (datetime), symbol (str), qty (signed), price (float), cost (float)

        Output columns:
        entry_time, exit_time, symbol, side, qty,
        entry_price, exit_price,
        gross_pnl, entry_cost, exit_cost, net_pnl,
        return_pct, hold_days
        """
        base_cols = [
            "entry_time", "exit_time", "symbol", "side", "qty",
            "entry_price", "exit_price",
            "gross_pnl", "entry_cost", "exit_cost", "net_pnl",
            "return_pct", "hold_days",
        ]

        if fills is None or fills.empty:
            return pd.DataFrame(columns=base_cols)

        f = fills.copy()

        # Accept prettified column names too (from _prepare_trades_table)
        if "qty" not in f.columns and "quantité" in f.columns:
            f["qty"] = pd.to_numeric(f["quantité"], errors="coerce")
        if "price" not in f.columns and "prix d'éxécution (open du jour)" in f.columns:
            f["price"] = pd.to_numeric(f["prix d'éxécution (open du jour)"], errors="coerce")

        # Normalize
        if "timestamp" not in f.columns or "symbol" not in f.columns:
            return pd.DataFrame(columns=base_cols)

        f["timestamp"] = pd.to_datetime(f["timestamp"], errors="coerce")
        f = f.dropna(subset=["timestamp"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        for c in ["qty", "price", "cost"]:
            if c in f.columns:
                f[c] = pd.to_numeric(f[c], errors="coerce")

        f = f.dropna(subset=["qty", "price"])
        if "cost" not in f.columns:
            f["cost"] = 0.0
        f["cost"] = f["cost"].fillna(0.0).astype(float)

        # FIFO lots per symbol
        # lot: qty_signed (float), price (float), time, entry_cost_total (float)
        lots: dict[str, list[dict]] = {}
        out_rows: list[dict] = []

        for _, row in f.iterrows():
            ts = row["timestamp"]
            sym = str(row["symbol"])
            qty = float(row["qty"])
            if qty == 0.0:
                continue

            price = float(row["price"])
            fill_cost_total = float(row["cost"])

            if sym not in lots:
                lots[sym] = []

            fill_abs = abs(qty)

            # Allocate fill cost proportional to the portion of this fill used
            def alloc_fill_cost(used_abs_qty: float) -> float:
                if fill_abs <= 0:
                    return 0.0
                return fill_cost_total * (float(used_abs_qty) / float(fill_abs))

            remaining_qty = qty  # signed float

            # Close opposite-direction lots FIFO
            while remaining_qty != 0.0 and lots[sym]:
                lot = lots[sym][0]
                lot_qty = float(lot["qty"])
                if lot_qty == 0.0:
                    lots[sym].pop(0)
                    continue

                # Same direction -> stop closing; remaining opens/increases
                if np.sign(lot_qty) == np.sign(remaining_qty):
                    break

                close_abs = min(abs(remaining_qty), abs(lot_qty))

                side = "LONG" if lot_qty > 0 else "SHORT"
                entry_price = float(lot["price"])
                exit_price = price

                # Gross PnL
                if side == "LONG":
                    gross = close_abs * (exit_price - entry_price)
                else:
                    gross = close_abs * (entry_price - exit_price)

                # Entry cost pro-rata from lot
                lot_abs_before = abs(lot_qty)
                entry_cost_part = float(lot["entry_cost"]) * (close_abs / lot_abs_before) if lot_abs_before > 0 else 0.0

                # Exit cost pro-rata from this fill
                exit_cost_part = alloc_fill_cost(close_abs)

                net = gross - entry_cost_part - exit_cost_part
                notional_entry = close_abs * entry_price
                ret_pct = (net / notional_entry) if notional_entry != 0 else np.nan

                hold_days = (pd.Timestamp(ts) - pd.Timestamp(lot["timestamp"])).days

                out_rows.append(
                    {
                        "entry_time": lot["timestamp"],
                        "exit_time": ts,
                        "symbol": sym,
                        "side": side,
                        "qty": float(close_abs),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_pnl": float(gross),
                        "entry_cost": float(entry_cost_part),
                        "exit_cost": float(exit_cost_part),
                        "net_pnl": float(net),
                        "return_pct": float(ret_pct) if np.isfinite(ret_pct) else np.nan,
                        "hold_days": int(hold_days),
                    }
                )

                # Reduce lot qty toward zero
                if lot_qty > 0:
                    lot["qty"] = lot_qty - close_abs
                else:
                    lot["qty"] = lot_qty + close_abs

                # Reduce lot entry cost by the portion consumed
                lot["entry_cost"] = float(lot["entry_cost"]) - float(entry_cost_part)

                # Reduce remaining fill qty toward zero
                if remaining_qty > 0:
                    remaining_qty -= close_abs
                else:
                    remaining_qty += close_abs

                # Drop depleted lot
                if abs(float(lot["qty"])) < 1e-12:
                    lots[sym].pop(0)

            # Any remaining opens a new lot
            if remaining_qty != 0.0:
                open_abs = abs(remaining_qty)
                entry_cost_for_open = alloc_fill_cost(open_abs)
                lots[sym].append(
                    {
                        "timestamp": ts,
                        "qty": float(remaining_qty),
                        "price": float(price),
                        "entry_cost": float(entry_cost_for_open),
                    }
                )

        # IMPORTANT: force schema even if no rows => prevents KeyError on merges
        ledger = pd.DataFrame(out_rows, columns=base_cols)

        # --- Optional audit columns from fills (sell clamp debugging) ---
        audit_cols = [
            "avg_entry_price_before",
            "avg_entry_price_after",
            "requested_sell_qty",
            "executed_sell_qty",
            "sell_clamp_reason",
        ]
        avail = [c for c in audit_cols if c in f.columns]

        # Only merge audits if we actually have closed trades (ledger non-empty)
        if avail and not ledger.empty:
            ff = f.copy()
            ff["side"] = ff.get("side", np.where(ff["qty"] > 0, "BUY", "SELL"))
            ff["side"] = ff["side"].astype(str).str.upper()
            sells = ff[ff["side"] == "SELL"][["timestamp", "symbol"] + avail].copy()
            sells = sells.rename(columns={"timestamp": "exit_time"})
            ledger = ledger.merge(sells, on=["exit_time", "symbol"], how="left")

        if ledger.empty:
            return ledger

        ledger = ledger.sort_values(["exit_time", "symbol"]).reset_index(drop=True)
        return ledger

    def _trade_performance_summary(self, ledger: pd.DataFrame) -> pd.DataFrame:
        if ledger is None or ledger.empty:
            return pd.DataFrame(index=[
                "Trades", "Win Rate", "Avg Net PnL", "Total Net PnL",
                "Avg Return %", "Profit Factor", "Avg Hold Days"
            ], data={"Value": [0, np.nan, np.nan, 0.0, np.nan, np.nan, np.nan]})

        net = ledger["net_pnl"].astype(float)
        wins = net[net > 0]
        losses = net[net < 0]

        trades = int(len(net))
        win_rate = float((net > 0).mean())
        avg_net = float(net.mean())
        total_net = float(net.sum())
        avg_ret = float(ledger["return_pct"].astype(float).mean())
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else np.inf
        avg_hold = float(ledger["hold_days"].astype(float).mean())

        return pd.DataFrame(
            {"Value": [trades, win_rate, avg_net, total_net, avg_ret, profit_factor, avg_hold]},
            index=["Trades", "Win Rate", "Avg Net PnL", "Total Net PnL",
                "Avg Return %", "Profit Factor", "Avg Hold Days"],
        )

