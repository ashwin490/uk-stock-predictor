"""Pure session-state decision for chart-free Streamlit market-hours refresh."""

def choose_auto_refresh(*, market_open: bool, in_entry_window: bool,
                        previous_market_open, previous_entry_window,
                        current_setting) -> bool:
    """Auto-start when the UK market or buy window opens, honour manual pauses
    until the next session boundary, and stop automatically after close."""
    if not market_open:
        return False
    if current_setting is None or previous_market_open is False:
        return True
    if in_entry_window and previous_entry_window is False:
        return True
    return bool(current_setting)
