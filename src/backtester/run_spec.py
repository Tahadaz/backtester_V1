# run_spec.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


def _canonical_json(obj: Any) -> str:
    """
    Deterministic JSON encoding (stable ordering) so hashing is consistent.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def run_id_from_spec(spec: Dict[str, Any]) -> str:
    """
    Hash a RunSpec dict into a stable run_id.
    """
    payload = _canonical_json(spec).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]  # short id is enough


def build_run_spec(
    *,
    source_key: str,
    symbols: List[str],
    start: Optional[str],
    end: Optional[str],
    include_windows: Any,
    exclude_windows: Any,
    interval: str,
    yf_period: str,
    yf_interval: str,
    yf_auto_adjust: bool,
    rank_metric: str,
    lb_opt_kinds: List[str],
    opt_method: str,
    n_trials: int,
    top_k: int,
    # portfolio-impacting params that MUST be part of the cache key:
    allow_short: bool,
    initial_cash: float,
    cooldown_bars: int,
    min_return_before_sell: float,
    cost_model: Dict[str, Any],
    volume_gate: Dict[str, Any],
    participation_cap: Dict[str, Any],
    # strategy domains (critical): store the active keys/domains per kind
    domains_by_kind: Dict[str, Any],
    # optional: version tag to invalidate old cached runs after code changes
    app_version: str = "v1",
) -> Dict[str, Any]:
    return {
        "app_version": app_version,
        "source_key": source_key,
        "symbols": symbols,
        "data": {
            "start": start,
            "end": end,
            "include_windows": include_windows,
            "exclude_windows": exclude_windows,
            "interval": interval,
            "yf_period": yf_period,
            "yf_interval": yf_interval,
            "yf_auto_adjust": yf_auto_adjust,
        },
        "optimization": {
            "rank_metric": rank_metric,
            "kinds": lb_opt_kinds,
            "method": opt_method,
            "n_trials": int(n_trials),
            "top_k": int(top_k),
            "domains_by_kind": domains_by_kind,
        },
        "portfolio": {
            "allow_short": bool(allow_short),
            "initial_cash": float(initial_cash),
            "cooldown_bars": int(cooldown_bars),
            "min_return_before_sell": float(min_return_before_sell),
            "cost_model": cost_model,
            "volume_gate": volume_gate,
            "participation_cap": participation_cap,
        },
    }
