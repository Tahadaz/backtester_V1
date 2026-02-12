# site_export.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


def ensure_results_tree(results_root: Path) -> None:
    (results_root / "runs").mkdir(parents=True, exist_ok=True)
    idx = results_root / "index.json"
    if not idx.exists():
        idx.write_text(json.dumps({"latest": None, "runs": []}, indent=2), encoding="utf-8")


def load_index(results_root: Path) -> Dict[str, Any]:
    idx = results_root / "index.json"
    if not idx.exists():
        return {"latest": None, "runs": []}
    return json.loads(idx.read_text(encoding="utf-8"))


def save_index(results_root: Path, index_obj: Dict[str, Any]) -> None:
    (results_root / "index.json").write_text(json.dumps(index_obj, indent=2), encoding="utf-8")


def write_plots_manifest(plots_dir: Path) -> None:
    """
    Your website expects plots/plots.json with {"plots":[...]}.
    We'll list all .html plot files in that dir.
    """
    html_files = sorted([p.name for p in plots_dir.glob("*.html")])
    payload = {"plots": [{"file": f, "label": f.replace("_", " ").replace(".html", "")} for f in html_files]}
    (plots_dir / "plots.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_store_to_run_folder(
    *,
    store: Dict[str, Any],
    run_id: str,
    run_spec: Dict[str, Any],
    results_root: Path,  # e.g. Path("results_website/assets/results")
    site_title: str = "Backtest Results",
    about_subtitle: str = "TA strategy optimization snapshots",
) -> Path:
    """
    Writes:
      results_root/runs/<run_id>/manifest.json
      results_root/runs/<run_id>/run_spec.json
      results_root/runs/<run_id>/stocks/<TICKER>/leaderboard.csv
      .../plots/*.html + plots.json
      .../ledgers/*.csv
      .../profile.json
      DONE
    Returns the run folder path.
    """
    ensure_results_tree(results_root)

    run_dir = results_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # manifest for website (run-scoped)
    tickers = sorted(store.keys())
    manifest = {
        "site_title": site_title,
        "about": {"subtitle": about_subtitle, "bullets": []},
        "stocks": [{"ticker": t, "name": t} for t in tickers],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "run_spec.json").write_text(json.dumps(run_spec, indent=2, default=str), encoding="utf-8")

    # write per-stock payloads (they already exist in your store)
    for ticker in tickers:
        payload = store[ticker]
        df_lb: pd.DataFrame = payload["leaderboard_df"]

        stock_dir = run_dir / "stocks" / ticker
        plots_dir = stock_dir / "plots"
        ledg_dir = stock_dir / "ledgers"
        stock_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        ledg_dir.mkdir(parents=True, exist_ok=True)

        # leaderboard
        (stock_dir / "leaderboard.csv").write_text(df_lb.to_csv(index=False), encoding="utf-8")

        # profile placeholder
        profile = {"ticker": ticker, "name": ticker, "summary": "", "key_points": []}
        (stock_dir / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

        # plots + ledgers are already present in your ZIP builder logic,
        # but currently only exist in-memory. For now we rely on YOU to call
        # this exporter right after you generate those plots/ledgers and place
        # them into these folders.
        #
        # To keep this step minimal, we just ensure plots.json exists after plots are written:
        # (you will write *.html before calling write_plots_manifest)
        #
        # We'll create plots.json even if empty:
        write_plots_manifest(plots_dir)

    # mark completion
    (run_dir / "DONE").write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")

    # update global index.json
    index_obj = load_index(results_root)
    index_obj["latest"] = run_id

    # prepend run metadata (avoid duplicates)
    runs = [r for r in index_obj.get("runs", []) if r.get("run_id") != run_id]
    runs.insert(0, {"run_id": run_id, "created_at": datetime.now().isoformat(timespec="seconds")})
    index_obj["runs"] = runs[:50]  # keep last 50 in registry
    save_index(results_root, index_obj)

    return run_dir
