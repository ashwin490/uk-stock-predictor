"""Portable, integrity-checked ALPHA-LSE data archive. Never includes secrets.

Usage (from a stopped Streamlit app):
    python alpha_backup.py backup
    python alpha_backup.py verify backups/alpha_backup_YYYYMMDD_HHMMSS.zip
    python alpha_backup.py restore backups/alpha_backup_YYYYMMDD_HHMMSS.zip --force

The Streamlit app can also call create_backup() while holding its DB_LOCK.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DATA_FILES = (
    "lse_market_data.duckdb",
    "models/ensemble_ranker.joblib",
    "models/ensemble_live.joblib",
    "models/retrain_state.json",
    "alpha_settings.json",
    "uk_universe.json",
    "aim_tickers.json",
    "extra_lse_tickers.json",
)
MANIFEST = "alpha_backup_manifest.json"
MAX_MEMBER_BYTES = 3 * 1024**3  # avoid accidental archive bombs on restore


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_backup(data_dir: str | Path, backup_dir: str | Path | None = None) -> Path:
    """Create an archive after checkpointing DuckDB; call under application DB lock.

    The archive is local. Copy it to another device or encrypted external disk.
    """
    data_dir = Path(data_dir).resolve()
    backup_dir = Path(backup_dir).resolve() if backup_dir is not None else data_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    db = data_dir / "lse_market_data.duckdb"
    if db.exists():
        import duckdb
        # On a separate process this will fail if the app has a conflicting open DB.
        # Never copy a potentially inconsistent, live WAL as if it were a complete DB.
        conn = duckdb.connect(str(db), read_only=False)
        try:
            conn.execute("CHECKPOINT")
        finally:
            conn.close()
    now = datetime.now(timezone.utc)
    output = backup_dir / f"alpha_backup_{now:%Y%m%d_%H%M%S_%f}.zip"
    if output.exists():
        raise FileExistsError(output)
    contents = {}
    with tempfile.NamedTemporaryFile(prefix="alpha_", suffix=".zip", dir=backup_dir, delete=False) as tmp:
        tmp_name = Path(tmp.name)
    try:
        with zipfile.ZipFile(tmp_name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for relative in DATA_FILES:
                file = data_dir / relative
                if not file.is_file() or file.is_symlink():
                    continue
                raw = file.read_bytes()
                if len(raw) > MAX_MEMBER_BYTES:
                    raise ValueError(f"File too large for a single backup: {relative}")
                archive.writestr(relative, raw)
                contents[relative] = {"sha256": _sha256(raw), "size": len(raw)}
            archive.writestr(MANIFEST, json.dumps({
                "format": "alpha-lse-learning-backup-v1",
                "created_utc": now.isoformat(), "files": contents,
                "note": "Contains historical paper trades and possibly serialized model objects. Treat as private."
            }, indent=2))
        verify_backup(tmp_name)
        os.replace(tmp_name, output)
        return output
    finally:
        if tmp_name.exists():
            tmp_name.unlink()


def verify_backup(archive_path: str | Path) -> dict:
    """Verify filenames, sizes and hashes before allowing restoration."""
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or MANIFEST not in names:
            raise ValueError("Backup manifest missing or duplicate entries")
        manifest = json.loads(archive.read(MANIFEST))
        if manifest.get("format") != "alpha-lse-learning-backup-v1":
            raise ValueError("Unknown backup format")
        members = manifest.get("files", {})
        if not isinstance(members, dict) or set(members) - set(DATA_FILES) or set(names) != set(members) | {MANIFEST}:
            raise ValueError("Backup includes unexpected files")
        for rel, meta in members.items():
            if rel.startswith("/") or ".." in Path(rel).parts:
                raise ValueError("Unsafe archive path")
            info = archive.getinfo(rel)
            if info.file_size > MAX_MEMBER_BYTES or info.file_size != meta.get("size"):
                raise ValueError(f"Invalid size for {rel}")
            if _sha256(archive.read(rel)) != meta.get("sha256"):
                raise ValueError(f"Checksum mismatch for {rel}")
        return manifest


def restore_backup(archive_path: str | Path, data_dir: str | Path, *, force: bool = False) -> list[str]:
    """Offline restore ONLY. Stop Streamlit first. Never execute untrusted joblib models."""
    manifest = verify_backup(archive_path)
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = list(manifest["files"])
    if not force and any((data_dir / name).exists() for name in paths):
        raise FileExistsError("Existing data found; back it up and use --force only when the app is stopped")
    with zipfile.ZipFile(archive_path, "r") as archive:
        with tempfile.TemporaryDirectory(prefix="alpha_restore_") as tmp:
            tmp_path = Path(tmp)
            for rel in paths:
                target = tmp_path / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(rel))
            # Existing files are kept in a timestamped rollback folder.
            rollback = data_dir / "pre_restore" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            for rel in paths:
                destination = data_dir / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    old = rollback / rel
                    old.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(destination, old)
                os.replace(tmp_path / rel, destination)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="ALPHA-LSE learning data backup / restore")
    parser.add_argument("action", choices=("backup", "verify", "restore"))
    parser.add_argument("archive", nargs="?", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("ALPHA_DATA_DIR", Path(__file__).resolve().parent)))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.action == "backup":
        print("Backed up:", create_backup(args.data_dir))
    elif args.action == "verify":
        if args.archive is None: parser.error("verify requires an archive path")
        result = verify_backup(args.archive)
        print("Verified:", len(result["files"]), "data files; created", result["created_utc"])
    elif args.action == "restore":
        if args.archive is None: parser.error("restore requires an archive path")
        print("Restored:", ", ".join(restore_backup(args.archive, args.data_dir, force=args.force)))


if __name__ == "__main__":
    main()
