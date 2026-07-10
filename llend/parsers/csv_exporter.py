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
    """Export *data* (list of dicts) to a CSV file.

    Called by ``ActionDispatcher`` as the ``export_csv`` action.
    """
    df = pd.DataFrame(data)
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding=encoding)
    return {
        "filename": str(path),
        "rows": len(df),
        "columns": list(df.columns),
    }
