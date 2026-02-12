# site_publish.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _plots_json_from_html_files(html_files: List[str]) -> Dict[str, Any]:
    return {
        "plots": [{"file": f, "label": f.replace("_", " ").replace(".html", "")} for f in sorted(html_files)]
    }


def export_run_to_results_site(
    *,
    store: Dict[str, Any],
    run_id: str,
    run_spec: Dict[str, Any],
    run_dir: Path,        # results_website/assets/results/runs/<run_id>
    results_root: Path,   # results_website/assets/results
) -> None:
    """
    Expects store[ticker] to contain at least:
      - leaderboard_df: pd.DataFrame
      - plots: Dict[str, str|bytes]  (filename -> HTML string or bytes)
      - ledgers: Dict[str, str|bytes] (filename -> CSV string or bytes)
    """

    tickers = sorted(store.keys())

    # Ensure run folder exists
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Write run-scoped manifest + run_spec
    manifest = {
        "site_title": "Backtester Results",
        "about": {"subtitle": "Optimization runs (cached)", "bullets": []},
        "stocks": [{"ticker": t, "name": t} for t in tickers],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "run_spec.json", run_spec)

    # 2) Per-stock artifacts
    for t in tickers:
        payload = store[t]

        stock_dir = run_dir / "stocks" / t
        plots_dir = stock_dir / "plots"
        ledg_dir  = stock_dir / "ledgers"

        # ✅ Create directories explicitly
        stock_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
        ledg_dir.mkdir(parents=True, exist_ok=True)

        # ---- REQUIRED: leaderboard ----
        df_lb = payload["leaderboard_df"]
        if not isinstance(df_lb, pd.DataFrame):
            raise TypeError(f"{t}: payload['leaderboard_df'] must be a DataFrame")
        _write_text(stock_dir / "leaderboard.csv", df_lb.to_csv(index=False))

        # ---- profile.json (minimal; you can enrich later) ----
        profile = {"ticker": t, "name": t, "summary": "", "key_points": []}
        _write_json(stock_dir / "profile.json", profile)

        # ---- plots ----
        plots_map = payload.get("plots", {})  # filename -> html str/bytes
        html_files: List[str] = []

        for fname, content in (plots_map or {}).items():
            out_path = plots_dir / fname
            if isinstance(content, bytes):
                _write_bytes(out_path, content)
            else:
                _write_text(out_path, str(content))
            if fname.lower().endswith(".html"):
                html_files.append(fname)

        # Always write plots.json even if empty
        _write_json(plots_dir / "plots.json", _plots_json_from_html_files(html_files))

        # ---- ledgers ----
        ledgers_map = payload.get("ledgers", {})  # filename -> csv str/bytes
        wrote_any_ledger = False

        for fname, content in (ledgers_map or {}).items():
            out_path = ledg_dir / fname
            if isinstance(content, bytes):
                _write_bytes(out_path, content)
            else:
                _write_text(out_path, str(content))
            wrote_any_ledger = True

        # If no ledgers exist, still leave a marker so the folder is present and visible
        if not wrote_any_ledger:
            _write_text(ledg_dir / "_EMPTY.txt", "No ledgers were exported for this ticker.\n")

    # 3) Update global index.json
    idx_path = results_root / "index.json"
    results_root.mkdir(parents=True, exist_ok=True)

    if idx_path.exists():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
    else:
        idx = {"latest": None, "runs": []}

    idx["latest"] = run_id
    runs = [r for r in idx.get("runs", []) if r.get("run_id") != run_id]
    runs.insert(0, {"run_id": run_id, "created_at": datetime.now().isoformat(timespec="seconds")})
    idx["runs"] = runs[:50]
    _write_json(idx_path, idx)

    # 4) DONE marker (write LAST, meaning “export completed successfully”)
    _write_text(run_dir / "DONE", datetime.now().isoformat(timespec="seconds"))

