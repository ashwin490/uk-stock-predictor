"""Explicit, backed-up removal of the obsolete options-only paper journal.

Never touches trade_journal / predictions. Never deletes on startup or as part
of a version migration. Run only after the signed-in user downloads the archive
and types the explicit confirmation.
"""
from __future__ import annotations
import io
import json
import zipfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import duckdb


def _remote_rows(client) -> list[dict]:
    if client is None:
        return []
    rows: list[dict] = []
    start = 0
    while True:
        response = client.table("options_journal").select("*").range(start, start + 999).execute()
        batch = getattr(response, "data", None)
        if batch is None:
            raise RuntimeError("Supabase options read was not confirmed")
        rows.extend(batch)
        if len(batch) < 1000:
            return rows
        start += len(batch)


def _local_rows(db_path: str) -> list[dict]:
    with duckdb.connect(db_path, read_only=True) as con:
        df = con.execute("SELECT * FROM daily_options_journal ORDER BY date_key").df()
        return df.to_dict(orient="records")


def make_options_archive(db_path: str, remote_client=None) -> tuple[bytes, dict]:
    """Produce user-downloadable ZIP (not a delete), includes all local/cloud options."""
    local = _local_rows(db_path)
    remote = _remote_rows(remote_client)
    if any(not str(r.get("date_key", "")).strip() for r in remote):
        raise RuntimeError("Cloud option history contains rows without date_key; no safe delete is possible")
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "local_options": len(local), "cloud_options": len(remote),
            "scope": "Options only, excludes all share and ETC trades"}
    payload = {"meta": meta, "local": local, "cloud": remote}
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("legacy_options_archive.json", json.dumps(payload, default=str, indent=2))
    return bio.getvalue(), meta


def retire_options(db_path: str, archived_zip: bytes, remote_client=None,
                   *, confirmation: str) -> dict:
    """Delete ONLY options, after validating the same rows were archived.

    If cloud deletion fails, local records are preserved for recovery. Some
    remote rows may already have been deleted; the backup holds them.
    """
    if confirmation != "DELETE OPTIONS":
        raise ValueError("Type DELETE OPTIONS to authorise this action")
    with zipfile.ZipFile(io.BytesIO(archived_zip), "r") as z:
        payload = json.loads(z.read("legacy_options_archive.json"))
    if not isinstance(payload, dict) or "local" not in payload or "cloud" not in payload:
        raise ValueError("Invalid options archive")
    # Compare against an original snapshot; fail rather than deleting trades
    # that arrived after the user created the archive.
    local_now = _local_rows(db_path)
    remote_now = _remote_rows(remote_client)
    # Compare complete rows as well as IDs, so a changed option P/L, status or
    # bid/ask mark cannot be silently discarded based on an outdated backup.
    def row_snapshot(rows):
        return json.dumps(sorted(rows, key=lambda x: str(x.get('date_key',''))),
                          sort_keys=True, default=str, allow_nan=True)
    if row_snapshot(local_now) != row_snapshot(payload['local']):
        raise RuntimeError("Local options changed after archive; prepare a fresh backup")
    if row_snapshot(remote_now) != row_snapshot(payload['cloud']):
        raise RuntimeError("Cloud options changed after archive; prepare a fresh backup")
    remote_ids = {str(r["date_key"]) for r in remote_now}
    if remote_client is not None:
        for date_key in sorted(remote_ids):
            remote_client.table("options_journal").delete().eq("date_key", date_key).execute()
        if _remote_rows(remote_client):
            raise RuntimeError("Cloud still contains options rows. Local history was not cleared")
    with duckdb.connect(db_path, read_only=False) as con:
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM daily_options_journal")
            count = con.execute("SELECT COUNT(*) FROM daily_options_journal").fetchone()[0]
            if count:
                raise RuntimeError("Could not clear local option history")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
    return {"local_deleted": len(local_now), "cloud_deleted": len(remote_now)}
