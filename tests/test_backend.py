"""Lifecycle tests for the ClickHouse state backend.

Runs against the ClickHouse instance at ``CH_TEST_URI``
(default ``clickhouse://default:clickhouse@localhost:8123/meltano``).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone

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
