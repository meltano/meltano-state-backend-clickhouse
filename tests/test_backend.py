"""Lifecycle tests for the ClickHouse state backend.

Runs against the ClickHouse instance at ``CH_TEST_URI``
(default ``clickhouse://default:clickhouse@localhost:8123/meltano``).
"""

from __future__ import annotations

import multiprocessing
import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import clickhouse_connect
import pytest
from meltano.core.state_store.base import MeltanoState

from meltano_state_backend_clickhouse.backend import (
    ClickhouseStateStoreManager,
    connection_params_from_uri,
)

URI = os.environ.get(
    "CH_TEST_URI",
    "clickhouse://default:clickhouse@localhost:8123/meltano",
)


@pytest.fixture
def manager() -> ClickhouseStateStoreManager:
    table = f"state_test_{uuid.uuid4().hex[:8]}"
    mgr = ClickhouseStateStoreManager(URI, table=table)
    yield mgr
    mgr.client.command(f"DROP TABLE IF EXISTS `{mgr.schema}`.`{table}`")
    mgr.client.command(f"DROP TABLE IF EXISTS `{mgr.schema}`.`{table}_lock`")
    mgr.close()


def test_connection_params_parsing() -> None:
    params = connection_params_from_uri("clickhouse://u:p@h:9000/db", secure=True)
    assert params["host"] == "h"
    assert params["port"] == 9000
    assert params["username"] == "u"
    assert params["password"] == "p"
    assert params["database"] == "db"
    assert params["secure"] is True


def test_set_get_overwrite(manager: ClickhouseStateStoreManager) -> None:
    sid = "dev:tap-to-target"
    manager.set(MeltanoState(state_id=sid, partial_state={}, completed_state={"v": 1}))
    assert manager.get(sid).completed_state == {"v": 1}
    # ReplacingMergeTree + FINAL: newest write wins.
    manager.set(MeltanoState(state_id=sid, partial_state={}, completed_state={"v": 2}))
    assert manager.get(sid).completed_state == {"v": 2}


def test_get_missing_returns_none(manager: ClickhouseStateStoreManager) -> None:
    assert manager.get("does-not-exist") is None


def test_state_ids_and_pattern(manager: ClickhouseStateStoreManager) -> None:
    manager.set(MeltanoState(state_id="dev:a", partial_state={}, completed_state={"v": 1}))
    manager.set(MeltanoState(state_id="prod:b", partial_state={}, completed_state={"v": 1}))
    assert set(manager.get_state_ids("*")) == {"dev:a", "prod:b"}
    assert set(manager.get_state_ids("dev:*")) == {"dev:a"}


def test_delete_and_clear_all(manager: ClickhouseStateStoreManager) -> None:
    manager.set(MeltanoState(state_id="dev:a", partial_state={}, completed_state={"v": 1}))
    manager.delete("dev:a")
    assert manager.get("dev:a") is None
    manager.set(MeltanoState(state_id="dev:b", partial_state={}, completed_state={"v": 1}))
    assert manager.clear_all() == 1
    assert list(manager.get_state_ids("*")) == []


def test_lock_round_trip_and_update(manager: ClickhouseStateStoreManager) -> None:
    sid = "dev:lock"
    with manager.acquire_lock(sid, retry_seconds=1):
        pass  # acquired and released without error
    # update() acquires the lock internally
    manager.update(MeltanoState(state_id=sid, partial_state={}, completed_state={"v": 9}))
    assert manager.get(sid).completed_state == {"v": 9}


def test_stale_lock_is_reclaimed(manager: ClickhouseStateStoreManager) -> None:
    # A lock left behind by a crashed holder (locked_at older than the stale
    # window) must be reclaimable — acquire_lock cleans it up synchronously
    # rather than relying on the TTL backstop (merge-time / best-effort).
    sid = "dev:stale"
    stale_at = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    manager.client.insert(
        f"{manager.schema}.{manager.table}_lock",
        [[sid, "dead-holder", stale_at]],
        column_names=["state_id", "lock_id", "locked_at"],
    )
    assert manager._lock_held_by(sid) == "dead-holder"  # noqa: SLF001
    with manager.acquire_lock(sid, retry_seconds=1):
        pass  # stale lock reclaimed, acquired and released without error


def _race_worker(worker_id: str, uri: str, table: str, state_id: str, events_table: str, hold_seconds: float) -> None:
    """Race a single worker into the shared lock; log enter/exit to events_table.

    Module-level so multiprocessing can pickle it. Each worker opens its own
    manager/client — connections are not fork-safe to share across processes.
    """
    mgr = ClickhouseStateStoreManager(uri, table=table)
    client = clickhouse_connect.get_client(**mgr.conn_params)
    try:
        with mgr.acquire_lock(state_id, retry_seconds=1):
            client.insert(events_table, [[worker_id, "enter"]], column_names=["worker", "kind"])
            time.sleep(hold_seconds)
            client.insert(events_table, [[worker_id, "exit"]], column_names=["worker", "kind"])
    finally:
        mgr.close()
        client.close()


def test_lock_serializes_concurrent_processes(manager: ClickhouseStateStoreManager) -> None:
    """Real concurrent processes racing the same state_id must never overlap.

    Regression test for a confirmed double-acquisition: before the
    settle-then-decide fix, acquire_lock's insert-then-immediately-verify
    pattern let two processes each independently observe themselves as the
    sole/winning row (FINAL resolves ReplacingMergeTree rows by the newest
    locked_at version, not by lock_id, so two near-simultaneous verifies can
    each only see their own not-yet-visible-to-the-other row). Reproduced on
    3/3 runs with 5 real OS processes before the fix; this test asserts it
    can no longer happen.
    """
    events_table = f"{manager.schema}.events_{uuid.uuid4().hex[:8]}"
    manager.client.command(
        f"CREATE TABLE {events_table} "
        "(worker String, kind String, ts DateTime64(6) DEFAULT now64(6)) "
        "ENGINE = MergeTree ORDER BY ts",
    )
    try:
        state_id = "dev:race"
        num_workers = 5
        hold_seconds = 1.0

        procs = [
            multiprocessing.Process(
                target=_race_worker,
                args=(f"w{i}", URI, manager.table, state_id, events_table, hold_seconds),
            )
            for i in range(num_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0

        rows = manager.client.query(
            f"SELECT worker, kind, ts FROM {events_table} ORDER BY ts",  # noqa: S608
        ).result_rows

        intervals: dict[str, dict[str, datetime]] = {}
        for worker, kind, ts in rows:
            intervals.setdefault(worker, {})[kind] = ts

        complete = sorted(
            (w, v["enter"], v["exit"]) for w, v in intervals.items() if "enter" in v and "exit" in v
        )
        assert len(complete) == num_workers

        for i in range(len(complete)):
            for j in range(i + 1, len(complete)):
                _, e1, x1 = complete[i]
                _, e2, x2 = complete[j]
                assert not (e1 < x2 and e2 < x1), (
                    f"critical sections overlapped: {complete[i]} vs {complete[j]}"
                )
    finally:
        manager.client.command(f"DROP TABLE IF EXISTS {events_table}")
