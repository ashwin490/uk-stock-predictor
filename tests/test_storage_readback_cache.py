"""Regressions for Supabase CDN stale reads following successful Storage updates."""
import json
from datetime import date

import pytest

import commodity_worker as worker
import share_worker as shares
from test_autonomous_worker import FakeBucket


def cloud_env(monkeypatch):
    monkeypatch.setenv('SUPABASE_URL', 'https://example.supabase.co')
    monkeypatch.setenv('SUPABASE_MODEL_SERVICE_KEY', 'private-test-key')
    monkeypatch.setenv('ALPHA_COMMODITY_ML_BUCKET', 'alpha-lse-models')


def test_private_fresh_get_uses_new_cache_nonce_and_private_authorization(monkeypatch):
    import requests
    cloud_env(monkeypatch)
    captures = []

    class Response:
        status_code = 200
        content = b'{"status":"ok"}'
        text = content.decode('utf-8')

    def mocked_get(url, *, headers, params, timeout):
        captures.append((url, headers, params, timeout))
        return Response()

    monkeypatch.setattr(requests, 'get', mocked_get)
    assert worker._fresh_private_json('research/shares/worker_health_v1.json') == {'status': 'ok'}
    assert worker._fresh_private_json('research/shares/worker_health_v1.json') == {'status': 'ok'}
    assert len(captures) == 2
    assert captures[0][2]['cacheNonce'] != captures[1][2]['cacheNonce']
    assert captures[0][0].endswith('/object/authenticated/alpha-lse-models/research/shares/worker_health_v1.json')
    assert 'private-test-key' not in captures[0][0]
    assert captures[0][1]['Authorization'] == 'Bearer private-test-key'


def test_successful_cloud_put_with_two_stale_reads_recovers(monkeypatch):
    cloud_env(monkeypatch)
    b = FakeBucket()
    old = {'schema': 1, 'last_mode': 'check'}
    b.files[shares.HEALTH] = worker.json_bytes(old)
    delays = {'count': 0}

    def fake_uncached(key):
        assert key == shares.HEALTH
        if b.updates and delays['count'] < 2:
            delays['count'] += 1
            return old
        return json.loads(b.files[key].decode('utf-8'))

    monkeypatch.setattr(worker, '_fresh_private_json', fake_uncached)
    monkeypatch.setattr(worker.sleep_time, 'sleep', lambda seconds: None)
    shares.health(b, 'audit', True, date(2026, 11, 24), {'status': 'ok', 'audited': 8})
    saved = json.loads(b.files[shares.HEALTH])
    assert delays['count'] == 2
    assert saved['last_mode'] == 'audit'
    assert saved['last_status'] == 'ok'
    assert saved['last_detail']['audited'] == 8


def test_permanently_mismatched_cloud_readback_still_fails(monkeypatch):
    cloud_env(monkeypatch)
    b = FakeBucket()
    old = {'schema': 1, 'version': 1}
    b.files['record.json'] = worker.json_bytes(old)
    monkeypatch.setattr(worker, '_fresh_private_json', lambda key: old)
    monkeypatch.setattr(worker.sleep_time, 'sleep', lambda seconds: None)
    with pytest.raises(RuntimeError, match='Cloud readback mismatch after retries'):
        worker.save_json(b, 'record.json', {'schema': 1, 'version': 2})
    # Failure is signalled; the worker does not falsely acknowledge the write.
    assert b.updates == 1


def test_cloud_missing_file_response_is_not_silent_for_non_missing_errors(monkeypatch):
    import requests
    cloud_env(monkeypatch)

    class Response:
        status_code = 400
        content = b'{"message":"Object not found"}'
        text = content.decode('utf-8')

    monkeypatch.setattr(requests, 'get', lambda *args, **kwargs: Response())
    assert worker.read_json(FakeBucket(), 'new.json', default=None, fresh=True) is None
    Response.status_code = 403
    Response.text = 'Forbidden'
    with pytest.raises(RuntimeError, match='HTTP 403'):
        worker.read_json(FakeBucket(), 'new.json', fresh=True)


def test_complete_share_audit_recovers_from_stale_health_get(monkeypatch):
    """Reproduce: audit JSON POST 200; health JSON PUT 200; first GETs stale."""
    from datetime import datetime, timezone
    cloud_env(monkeypatch)
    b = FakeBucket()
    b.files[shares.HEALTH] = worker.json_bytes({'schema': 1, 'last_mode': 'check'})
    delays = {'count': 0}

    def remote(key):
        if key == shares.HEALTH and b.updates and delays['count'] < 2:
            delays['count'] += 1
            return {'schema': 1, 'last_mode': 'check'}
        content = b.files.get(key)
        return json.loads(content) if content is not None else None

    monkeypatch.setattr(worker, '_fresh_private_json', remote)
    monkeypatch.setattr(worker.sleep_time, 'sleep', lambda seconds: None)

    class Response:
        data = []
    class Query:
        def select(self, *args): return self
        def range(self, *args): return self
        def execute(self): return Response()
    class Client:
        def table(self, name):
            assert name == 'predictions'
            return Query()

    report = shares.accounting(b, Client(), datetime(2026, 11, 24, 16, 8, tzinfo=timezone.utc))
    assert 'audited' in report
    assert json.loads(b.files[shares.HEALTH])['last_mode'] == 'audit'
    assert json.loads(b.files[shares.AUDIT])['audited'] == 0
    assert delays['count'] == 2
