"""CSV exporter — wraps pandas ``DataFrame.to_csv`` for the ``export_csv`` action.

``DataFrame.to_csv`` is an instance method; ``ActionDispatcher`` needs a
plain function.  This module provides one.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


async def export_csv(
    data: list[dict[str, Any]],
    filename: str,
    index: bool = False,
    encoding: str = "utf-8-sig",
) -> dict[str, Any]:
    """Export *data* to a CSV file.

    Called by ``ActionDispatcher`` as the ``export_csv`` action.
    Handles multiple input formats: list of dicts, single dict, list of lists.
    """
    # Normalise data into a list-of-dicts format for DataFrame
    try:
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            # Single row or dict-of-columns
            df = pd.DataFrame([data])
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            df = pd.DataFrame(data[1:], columns=data[0] if data[0] else None)
        else:
            df = pd.DataFrame(data)
    except Exception as exc:
        logger.warning("Failed to create DataFrame: %s — trying fallback", exc)
        df = pd.DataFrame({"data": [str(data)]})

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding=encoding)
    return {
        "filename": str(path),
        "rows": len(df),
        "columns": list(df.columns),
    }
