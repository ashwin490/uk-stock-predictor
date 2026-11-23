"""Pure decisions for Streamlit market-hours refresh.

Refresh defaults ON whenever a browser session is opened during a UK trading
session. An *explicit* in-session pause is respected until the next market or
entry-window opening; it must not leak into a newly opened session.
"""


def choose_auto_refresh(*, market_open: bool, in_entry_window: bool,
                        previous_market_open, previous_entry_window,
                        current_setting, manually_paused: bool = False,
                        session_initialized: bool = False) -> bool:
    """Return whether the auto-refresh toggle should be enabled.

    The market-open predicate is authoritative (includes holidays and half-days).
    Entry-window changes re-enable refresh after a deliberate manual pause;
    automated scanning can remain ON outside the buy window to monitor trades.
    """
    if not market_open:
        return False
    if not session_initialized:
        return True
    if previous_market_open is False or (in_entry_window and previous_entry_window is False):
        return True
    if manually_paused:
        return False
    return True
