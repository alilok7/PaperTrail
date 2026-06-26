"""Lightweight progress reporting for long-running ingestion/embedding work.

A :data:`ProgressFn` is a sink the front-ends pass down so they can show live status
(spinner labels, progress bars, rate-limit waits) while ingestion runs. It is optional
everywhere — ``None`` means "report nothing" — so the library stays usable headless.
"""

from __future__ import annotations

from typing import Callable, Optional

# (human-readable message, fraction-complete in [0, 1] or None when indeterminate).
ProgressFn = Callable[[str, Optional[float]], None]


def report(on_progress: ProgressFn | None, message: str, fraction: float | None = None) -> None:
    """Send one progress update, swallowing any error in the sink.

    A misbehaving UI callback must never crash the work it is only observing.
    """
    if on_progress is None:
        return
    try:
        on_progress(message, fraction)
    except Exception:  # noqa: BLE001 - progress reporting is best-effort
        pass
