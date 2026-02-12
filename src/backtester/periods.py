from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd

Window = Tuple[str, str]


# -----------------------------
# Market-wide (MASI) periods
# -----------------------------
# These are intended as *research regimes* (vol/liquidity/risk-premium shifts).
MASI_PERIODS: Dict[str, Window] = {
    "Pre-GFC (2005-2007)": ("2005-01-01", "2007-12-31"),
    "GFC spillover (2008-2009)": ("2008-01-01", "2009-12-31"),
    "Euro/Arab Spring risk (2010-2012)": ("2010-01-01", "2012-12-31"),
    "Normalization (2013-2019)": ("2013-01-01", "2019-12-31"),
    "COVID shock & aftershocks (2020-2021)": ("2020-01-01", "2021-12-31"),
    "Inflation / rates shock (2022-2023)": ("2022-01-01", "2023-12-31"),
    "Post-2024 cycle (2024-2026)": ("2024-01-01", "2026-12-31"),
}

# -----------------------------
# Stock-specific overlay periods
# -----------------------------
# IAM examples. Expand as you research / document additional events.
STOCK_PERIODS: Dict[str, Dict[str, Window]] = {
    "IAM": {
        # Ownership / control era (Etisalat/Vivendi transaction closes early 2014; use a wide window)
        "Control transition (2013-2014)": ("2013-01-01", "2014-12-31"),

        # COVID + regulatory pressure era (wide, captures microstructure & legal/regulatory narratives)
        "COVID + regulatory pressure (2020-2021)": ("2020-01-01", "2021-12-31"),

        # Inwi litigation overhang (wide; you can refine later)
        "Inwi litigation overhang (2024-01-29 to 2025-03-01)": ("2024-01-29", "2025-03-01"),

        # Governance change era (effective 2025-03-01 in public reporting)
        "Post governance change (2025-03-01+)": ("2025-03-01", "2026-12-31"),
    }
}


def windows_from_labels(registry: Dict[str, Window], labels: List[str]) -> List[Window]:
    return [registry[lbl] for lbl in labels if lbl in registry]


def intersect_two_windows(a: Window, b: Window) -> Optional[Window]:
    a0, a1 = pd.Timestamp(a[0]), pd.Timestamp(a[1])
    b0, b1 = pd.Timestamp(b[0]), pd.Timestamp(b[1])
    start = max(a0, b0)
    end = min(a1, b1)
    if start > end:
        return None
    return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


def intersect_windows(A: List[Window], B: List[Window]) -> List[Window]:
    out: List[Window] = []
    for a in A:
        for b in B:
            w = intersect_two_windows(a, b)
            if w is not None:
                out.append(w)
    return out


def normalize_windows(windows: List[Window]) -> List[Window]:
    # Sort and merge overlaps (keeps engine fast and results stable)
    if not windows:
        return []
    xs = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in windows]
    xs.sort(key=lambda t: t[0])
    merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur_a, cur_b = xs[0]
    for a, b in xs[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))
    return [(a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d")) for a, b in merged]
