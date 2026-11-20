"""Safety regression checks: cloud acknowledgements and years-long portable archives."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

from test_local_first import functions

ROOT = Path(__file__).resolve().parents[1]


def backup_module():
    spec = importlib.util.spec_from_file_location('final_backup', ROOT / 'alpha_backup.py')
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_archive_streams_data_without_path_read_bytes(tmp_path, monkeypatch):
    backup = backup_module()
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'alpha_settings.json').write_bytes(b'a' * (2 * 1024 * 1024 + 3))
    (data / 'models').mkdir()
    (data / 'models' / 'retrain_state.json').write_text('{"n": 9}')

    def prohibit_whole_file_read(*_args, **_kwargs):
        raise AssertionError('backup may not materialize a data file in RAM')

    monkeypatch.setattr(Path, 'read_bytes', prohibit_whole_file_read)
    path = backup.create_backup(data)
    manifest = backup.verify_backup(path)
    assert manifest['files']['alpha_settings.json']['size'] == 2 * 1024 * 1024 + 3
    assert 'models/retrain_state.json' in manifest['files']


def test_restore_rolls_back_all_files_if_second_replace_fails(tmp_path, monkeypatch):
    backup = backup_module()
    old = tmp_path / 'old'
    old.mkdir()
    (old / 'alpha_settings.json').write_text('new-settings')
    (old / 'models').mkdir()
    (old / 'models' / 'retrain_state.json').write_text('new-state')
    archive = backup.create_backup(old)

    target = tmp_path / 'target'
    target.mkdir()
    (target / 'alpha_settings.json').write_text('original-settings')
    (target / 'models').mkdir()
    (target / 'models' / 'retrain_state.json').write_text('original-state')

    replace = os.replace
    failures = {'n': 0}

    def fail_only_model_install(src, dest):
        if (str(src).endswith('models/retrain_state.json') and
                'alpha_restore_' in str(src) and failures['n'] == 0):
            failures['n'] += 1
            raise OSError('simulated storage error midway through restore')
        return replace(src, dest)

    monkeypatch.setattr(os, 'replace', fail_only_model_install)
    with pytest.raises(OSError, match='storage error'):
        backup.restore_backup(archive, target, force=True)
    assert (target / 'alpha_settings.json').read_text() == 'original-settings'
    assert (target / 'models' / 'retrain_state.json').read_text() == 'original-state'


@pytest.mark.parametrize('acknowledgement,expected', [(None, False), ([], False), ([{'id': 1}], True)])
def test_cloud_update_requires_positive_database_acknowledgement(acknowledgement, expected):
    class Query:
        def update(self, payload):
            return self
        def eq(self, key, value):
            return self
        def execute(self):
            return SimpleNamespace(data=acknowledgement)

    cloud = SimpleNamespace(table=lambda _: Query())
    state = {}
    fn = functions('cloud_update', STORAGE_MODE='cloud', supabase=cloud,
                   st=SimpleNamespace(session_state=state),
                   record_db_error=lambda label, error: state.update(error=type(error).__name__))['cloud_update']
    result = fn('predictions', {'ticker':'BARC.L'}, {'status':'CLOSED'}, 'Test')
    assert result is expected
    assert ('pending_cloud_sync' in state) is (not expected)


def test_authentication_prefers_argon2_and_preserves_legacy_sha256():
    from argon2 import PasswordHasher
    import hashlib
    fn = functions('verify_login_password',
                   )['verify_login_password']
    pwd = 'secure-example-password-not-a-real-login'
    argon = PasswordHasher().hash(pwd)
    old = hashlib.sha256(pwd.encode('utf-8')).hexdigest()
    assert fn(pwd, argon2id_hash=argon)
    assert not fn('wrong', argon2id_hash=argon)
    assert fn(pwd, legacy_sha256_hash=old)
    assert not fn('wrong', legacy_sha256_hash=old)
    # Never fall back to the weaker legacy secret if Argon2id was supplied.
    assert not fn(pwd, argon2id_hash='$argon2id$broken', legacy_sha256_hash=old)
