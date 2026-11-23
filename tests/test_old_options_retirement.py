"""No-real-database tests for safeguards around deletion of obsolete options.

Uses an in-memory fake DuckDB connector because CI dependency tests can be run
without cloud credentials. This module must never delete the shares table.
"""
import sys
import types
import json
import io
import zipfile
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault('duckdb', types.SimpleNamespace(connect=None))
import option_retirement as retire


class MockCloud:
    def __init__(self, rows):
        self.rows = [dict(x) for x in rows]
    def table(self, table):
        assert table == 'options_journal'  # never touches predictions
        return MockQuery(self)


class MockQuery:
    def __init__(self, cloud):
        self.cloud=cloud
        self.lo=0
        self.hi=999
        self.date=None
        self.is_delete=False
    def select(self, cols):
        assert cols=='*'
        return self
    def range(self, lo, hi):
        self.lo,self.hi=lo,hi
        return self
    def delete(self):
        self.is_delete=True
        return self
    def eq(self, key, value):
        assert key=='date_key'
        self.date=value
        return self
    def execute(self):
        if self.is_delete:
            self.cloud.rows=[x for x in self.cloud.rows if x['date_key'] != self.date]
            return types.SimpleNamespace(data=[])
        return types.SimpleNamespace(data=self.cloud.rows[self.lo:self.hi+1])


def test_archive_contains_both_cloud_and_local_and_is_downloadable(monkeypatch):
    local=[{'date_key':'2026-11-18', 'status':'LOSS', 'pnl_pct':-22.2}]
    cloud=MockCloud([{'date_key':'2026-11-20', 'status':'ACTIVE', 'pnl_pct':-5.2}])
    monkeypatch.setattr(retire, '_local_rows',lambda _:local)
    raw, meta=retire.make_options_archive('unused.db',cloud)
    assert meta['local_options']==1 and meta['cloud_options']==1
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        obj=json.loads(z.read('legacy_options_archive.json'))
    assert obj['local']==local
    assert obj['cloud']==cloud.rows
    assert cloud.rows  # archive must not delete


def test_archive_refuses_changes_before_deletion(monkeypatch):
    rows=[{'date_key':'2026-11-18', 'status':'ACTIVE', 'pnl_pct':-5}]
    cloud=MockCloud([])
    monkeypatch.setattr(retire,'_local_rows',lambda _:rows)
    raw,_=retire.make_options_archive('unused.db', cloud)
    rows[0]['pnl_pct']=-80
    with pytest.raises(RuntimeError,match='changed after archive'):
        retire.retire_options('unused.db',raw,cloud,confirmation='DELETE OPTIONS')
    assert rows and cloud.rows==[]


def test_failed_cloud_deletion_does_not_clear_local_archive(monkeypatch):
    rows=[{'date_key':'2026-11-18', 'status':'ACTIVE'}]
    cloud=MockCloud([{'date_key':'2026-11-19', 'status':'ACTIVE'}])
    monkeypatch.setattr(retire,'_local_rows',lambda _:rows)
    raw,_=retire.make_options_archive('unused.db',cloud)
    class BrokenCloud(MockCloud):
        def table(self, table):
            q=super().table(table)
            if table=='options_journal':
                q.execute=lambda: (_ for _ in ()).throw(RuntimeError('write down'))
            return q
    broken=BrokenCloud(cloud.rows)
    with pytest.raises(RuntimeError,match='write down'):
        retire.retire_options('unused.db',raw,broken,confirmation='DELETE OPTIONS')
    assert rows
