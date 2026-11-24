"""Regression checks for existing Supabase state file and safer ML retries."""
from pathlib import Path
import pytest
from commodity_ml import CommodityMLStore, MODEL_SCHEMA


class StorageConflict(Exception):
    def __init__(self):
        self.status_code = 409
        super().__init__('The resource already exists')


class ExistingFileBucket:
    def __init__(self):
        self.data = b'OLD MODEL STATE'
        self.upload_calls = 0
        self.update_calls = 0
        self.read_calls = 0

    def upload(self, path, payload, file_options=None):
        self.upload_calls += 1
        raise StorageConflict()

    def update(self, path, payload, file_options=None):
        self.update_calls += 1
        assert path == 'commodity_ml_champion_v1.json'
        assert file_options['content-type'] == 'application/json'
        self.data = payload

    def download(self, path):
        self.read_calls += 1
        return self.data


def test_existing_model_file_can_be_updated_without_deleting(tmp_path):
    bucket = ExistingFileBucket()
    store = CommodityMLStore(tmp_path / 'ephemeral.json', cloud_mode=True, bucket=bucket)
    new_state = {'schema': MODEL_SCHEMA, 'last_attempt_day': '2026-11-24',
                 'last_evaluation': {'test_count': 1148}}
    store.save(new_state)
    assert store.load() == new_state
    assert bucket.upload_calls == 1
    assert bucket.update_calls == 1
    assert bucket.read_calls == 2
    assert not (tmp_path / 'ephemeral.json').exists()


def test_only_duplicate_error_can_trigger_update(tmp_path):
    class ForbiddenBucket(ExistingFileBucket):
        def upload(self, path, payload, file_options=None):
            raise PermissionError('HTTP 403 forbidden')
    bucket = ForbiddenBucket()
    with pytest.raises(RuntimeError, match='not activated'):
        CommodityMLStore(tmp_path / 'ephemeral.json', cloud_mode=True, bucket=bucket).save({'schema': MODEL_SCHEMA})
    assert bucket.update_calls == 0
    assert bucket.data == b'OLD MODEL STATE'


def test_readback_must_match_after_update(tmp_path):
    class StaleReadbackBucket(ExistingFileBucket):
        def download(self, path):
            return b'OLD MODEL STATE'
    bucket = StaleReadbackBucket()
    with pytest.raises(RuntimeError, match='not activated'):
        CommodityMLStore(tmp_path / 'ephemeral.json', cloud_mode=True, bucket=bucket).save({'schema': MODEL_SCHEMA})
    assert bucket.update_calls == 1


def test_app_retries_after_failure_and_does_not_publish_private_errors():
    source = (Path(__file__).resolve().parents[1] / 'app.py').read_text()
    assert "st.session_state['commodity_ml_retry_at'] = time.time() + 900" in source
    assert "st.session_state['commodity_ml_checked_day'] = today.isoformat()" in source
    assert "f'Commodity ML update failed while {stage} ({label}). '" in source
    assert "LOGGER.warning('Commodity ML failure at %s (%s)', stage, label)" in source
    # Every successful candidate must still persist before activating it.
    assert 'store.save(proposed)  # MUST persist before in-session activation.' in source
