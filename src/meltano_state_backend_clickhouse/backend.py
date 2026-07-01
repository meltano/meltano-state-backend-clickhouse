"""StateStoreManager for ClickHouse state backend.

ClickHouse is an OLAP column store and intentionally lacks the primitives a
transactional state store would normally rely on:

* no row-level ``UPDATE`` — we model the state table as a ``ReplacingMergeTree``
  keyed by ``state_id`` and read it back with ``FINAL`` so the newest row
  (highest ``updated_at`` version) wins;
* no ``INSERT`` uniqueness/conflict error — so the lock table cannot rely on a
  primary-key violation the way the MSSQL/Postgres backends do. We implement a
  *best-effort* advisory lock (insert-then-verify with stale-lock cleanup).
  Meltano namespaces state per pipeline via ``--state-id-suffix`` and the
  platform serialises runs per state id, so contention is low in practice.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import clickhouse_connect
from meltano.core.error import MeltanoError
from meltano.core.setting_definition import SettingDefinition, SettingKind
from meltano.core.state_store.base import (
    MeltanoState,
    MissingStateBackendSettingsError,
    StateIDLockedError,
    StateStoreManager,
)
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from clickhouse_connect.driver.client import Client


DEFAULT_TABLE_NAME = "state"
DEFAULT_SCHEMA_NAME = "meltano"
DEFAULT_PORT = 8123
LOCK_TIMEOUT_SECONDS = 30
STALE_LOCK_SECONDS = 300  # 5 minutes

logger = logging.getLogger(__name__)


class ClickhouseStateBackendError(MeltanoError):
    """Base error for the ClickHouse state backend."""


CLICKHOUSE_HOST = SettingDefinition(
    name="state_backend.clickhouse.host",
    label="ClickHouse Host",
    description="ClickHouse server hostname",
    kind=SettingKind.STRING,
    env_specific=True,
)

CLICKHOUSE_PORT = SettingDefinition(
    name="state_backend.clickhouse.port",
    label="ClickHouse Port",
    description="ClickHouse HTTP interface port",
    kind=SettingKind.INTEGER,
    env_specific=True,
)

CLICKHOUSE_DATABASE = SettingDefinition(
    name="state_backend.clickhouse.database",
    label="ClickHouse Database",
    description="ClickHouse database name",
    kind=SettingKind.STRING,
    env_specific=True,
)

CLICKHOUSE_USER = SettingDefinition(
    name="state_backend.clickhouse.user",
    label="ClickHouse User",
    description="ClickHouse username",
    kind=SettingKind.STRING,
    env_specific=True,
)

CLICKHOUSE_PASSWORD = SettingDefinition(
    name="state_backend.clickhouse.password",
    label="ClickHouse Password",
    description="ClickHouse password",
    kind=SettingKind.STRING,
    sensitive=True,
    env_specific=True,
)

CLICKHOUSE_SECURE = SettingDefinition(
    name="state_backend.clickhouse.secure",
    label="ClickHouse Secure (TLS)",
    description="Connect to ClickHouse over HTTPS/TLS",
    kind=SettingKind.BOOLEAN,
    env_specific=True,
)

CLICKHOUSE_SCHEMA = SettingDefinition(
    name="state_backend.clickhouse.schema",
    label="ClickHouse Database/Schema",
    description="ClickHouse database used for state storage (default: meltano)",
    kind=SettingKind.STRING,
    env_specific=True,
)

CLICKHOUSE_TABLE = SettingDefinition(
    name="state_backend.clickhouse.table",
    label="ClickHouse Table",
    description="ClickHouse table name for state storage (default: state)",
    kind=SettingKind.STRING,
    env_specific=True,
)


def connection_params_from_uri(
    uri: str,
    *,
    host: str | None = None,
    port: int | None = None,
    database: str | None = None,
    user: str | None = None,
    password: str | None = None,
    secure: bool | None = None,
) -> dict[str, Any]:
    """Build clickhouse-connect client kwargs from a URI plus optional overrides.

    Args:
        uri: ClickHouse state backend URI (``clickhouse://user:pass@host:port/db``).
        host: Override hostname.
        port: Override port.
        database: Override database name.
        user: Override username.
        password: Override password.
        secure: Override TLS flag.

    Returns:
        A dict of kwargs suitable for ``clickhouse_connect.get_client``.

    Raises:
        MissingStateBackendSettingsError: If a required parameter is missing.
    """
    parsed = urlparse(uri)

    params: dict[str, Any] = {}

    if parsed.hostname:
        params["host"] = parsed.hostname
    if parsed.port:
        params["port"] = parsed.port
    if parsed.username:
        params["username"] = unquote(parsed.username)
    if parsed.password:
        params["password"] = unquote(parsed.password)
    if parsed.path and parsed.path.lstrip("/"):
        params["database"] = parsed.path.lstrip("/")

    # Apply explicit overrides
    if host:
        params["host"] = host
    if port is not None:
        params["port"] = port
    if user:
        params["username"] = user
    if password:
        params["password"] = password
    if database:
        params["database"] = database
    if secure is not None:
        params["secure"] = secure

    if not params.get("username"):
        msg = "ClickHouse user is required"
        raise MissingStateBackendSettingsError(msg)
    if not params.get("database"):
        msg = "ClickHouse database is required"
        raise MissingStateBackendSettingsError(msg)

    params.setdefault("host", "localhost")
    params.setdefault("port", DEFAULT_PORT)
    params.setdefault("password", "")

    return params


class ClickhouseStateStoreManager(StateStoreManager):
    """State backend for ClickHouse."""

    @property
    @override
    def label(self) -> str:
        """Return a human-readable label for this backend."""
        return "ClickHouse"  # pragma: no cover

    def __init__(
        self,
        uri: str,
        *,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        secure: bool | None = None,
        schema: str | None = None,
        table: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the ClickhouseStateStoreManager.

        Args:
            uri: The state backend URI (``clickhouse://user:pass@host:port/db``).
            host: ClickHouse hostname override.
            port: ClickHouse port override.
            database: ClickHouse database override.
            user: ClickHouse username override.
            password: ClickHouse password override.
            secure: Connect over TLS.
            schema: ClickHouse database used for state (default: meltano).
            table: State table name (default: state).
            kwargs: Additional keyword args passed to the parent.
        """
        super().__init__(**kwargs)
        self.uri = uri
        self.conn_params = connection_params_from_uri(
            uri,
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            secure=secure,
        )

        self.schema = schema or DEFAULT_SCHEMA_NAME
        self.table = table or DEFAULT_TABLE_NAME
        self.state_table = f"`{self.schema}`.`{self.table}`"
        self.lock_table = f"`{self.schema}`.`{self.table}_lock`"

        self._client: Client | None = None
        self._ensure_tables()

    @property
    def client(self) -> Client:
        """Return a cached clickhouse-connect client.

        Returns:
            A clickhouse-connect client.
        """
        if self._client is None:
            self._client = clickhouse_connect.get_client(**self.conn_params)
        return self._client

    @client.setter
    def client(self, value: Client) -> None:
        """Set the clickhouse-connect client (for testing/mocking)."""
        self._client = value

    def _ensure_tables(self) -> None:
        """Create the state database and the state/lock tables if absent."""
        self.client.command(f"CREATE DATABASE IF NOT EXISTS `{self.schema}`")
        # ReplacingMergeTree(updated_at) keeps the newest row per state_id; reads
        # use FINAL to collapse superseded versions.
        self.client.command(
            f"""
            CREATE TABLE IF NOT EXISTS {self.state_table} (
                state_id String,
                partial_state Nullable(String),
                completed_state Nullable(String),
                updated_at DateTime64(3) DEFAULT now64(3)
            ) ENGINE = ReplacingMergeTree(updated_at)
            ORDER BY state_id
            """,  # noqa: S608
        )
        # A TTL on locked_at is a backstop: if a lock holder dies without
        # releasing, the row is reaped by background merges after the stale
        # window even if no one else contends the same state_id (which is what
        # otherwise triggers the active DELETE cleanup). It does not replace the
        # active cleanup — TTL is merge-time / best-effort, so acquire_lock still
        # clears stale rows synchronously for liveness within the window.
        self.client.command(
            f"""
            CREATE TABLE IF NOT EXISTS {self.lock_table} (
                state_id String,
                lock_id String,
                locked_at DateTime DEFAULT now()
            ) ENGINE = ReplacingMergeTree(locked_at)
            ORDER BY state_id
            TTL locked_at + INTERVAL {STALE_LOCK_SECONDS} SECOND
            """,  # noqa: S608
        )

    @override
    def set(self, state: MeltanoState) -> None:
        """Insert (and supersede) the state row for the given state id.

        Args:
            state: the state to set.
        """
        partial_json = json.dumps(state.partial_state) if state.partial_state else None
        completed_json = (
            json.dumps(state.completed_state) if state.completed_state else None
        )
        self.client.insert(
            f"{self.schema}.{self.table}",
            [[state.state_id, partial_json, completed_json]],
            column_names=["state_id", "partial_state", "completed_state"],
        )

    @override
    def get(self, state_id: str) -> MeltanoState | None:
        """Get the merged state for the given state id.

        Args:
            state_id: the name of the job to get state for.

        Returns:
            The current state, or None if not found.
        """
        result = self.client.query(
            f"SELECT partial_state, completed_state FROM {self.state_table} FINAL "  # noqa: S608
            "WHERE state_id = {state_id:String} LIMIT 1",
            parameters={"state_id": state_id},
        )
        if not result.result_rows:
            return None

        partial_state, completed_state = result.result_rows[0]
        return MeltanoState(
            state_id=state_id,
            partial_state=json.loads(partial_state) if partial_state else {},
            completed_state=json.loads(completed_state) if completed_state else {},
        )

    @override
    def delete(self, state_id: str) -> None:
        """Delete state for the given state id.

        Args:
            state_id: the state_id to clear state for.
        """
        self.client.command(
            f"DELETE FROM {self.state_table} WHERE state_id = {{state_id:String}}",  # noqa: S608
            parameters={"state_id": state_id},
        )

    @override
    def clear_all(self) -> int:
        """Clear all states.

        Returns:
            The number of states cleared.
        """
        result = self.client.query(
            f"SELECT count() FROM {self.state_table} FINAL",  # noqa: S608
        )
        count = int(result.result_rows[0][0]) if result.result_rows else 0
        self.client.command(f"TRUNCATE TABLE {self.state_table}")
        return count

    @override
    def get_state_ids(self, pattern: str | None = None) -> Iterable[str]:
        """Get all state ids, optionally filtered by a glob pattern.

        Args:
            pattern: glob-style pattern to filter by.

        Returns:
            An iterable of state ids.
        """
        if pattern and pattern != "*":
            sql_pattern = pattern.replace("*", "%").replace("?", "_")
            result = self.client.query(
                f"SELECT DISTINCT state_id FROM {self.state_table} FINAL "  # noqa: S608
                "WHERE state_id LIKE {pattern:String}",
                parameters={"pattern": sql_pattern},
            )
        else:
            result = self.client.query(
                f"SELECT DISTINCT state_id FROM {self.state_table} FINAL",  # noqa: S608
            )
        return [str(row[0]) for row in result.result_rows]

    @override
    def close(self) -> None:
        """Close the ClickHouse client if it has been opened."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def _cleanup_stale_locks(self) -> None:
        """Remove locks older than STALE_LOCK_SECONDS."""
        self.client.command(
            f"DELETE FROM {self.lock_table} "  # noqa: S608
            f"WHERE locked_at < now() - {STALE_LOCK_SECONDS}",
        )

    def _lock_held_by(self, state_id: str) -> str | None:
        """Return the winning lock_id for a state id, or None if unlocked."""
        result = self.client.query(
            f"SELECT lock_id FROM {self.lock_table} FINAL "  # noqa: S608
            "WHERE state_id = {state_id:String} ORDER BY lock_id LIMIT 1",
            parameters={"state_id": state_id},
        )
        if not result.result_rows:
            return None
        return str(result.result_rows[0][0])

    @override
    @contextmanager
    def acquire_lock(
        self,
        state_id: str,
        *,
        retry_seconds: int = 1,
    ) -> Generator[None, None, None]:
        """Acquire a best-effort advisory lock for the given state id.

        ClickHouse has no insert-conflict primitive, so this inserts a candidate
        lock row and then verifies it owns the lock (lowest ``lock_id`` wins ties).
        Stale locks are cleaned up first.

        Args:
            state_id: the state_id to lock.
            retry_seconds: seconds to wait between retries.

        Yields:
            None

        Raises:
            StateIDLockedError: if the lock cannot be acquired within the timeout.
        """
        lock_id = str(uuid.uuid4())
        seconds_waited = 0.0
        acquired = False

        while seconds_waited < LOCK_TIMEOUT_SECONDS:
            holder = self._lock_held_by(state_id)
            if holder is not None:
                # A lock row exists — it may be stale. Only now pay for the
                # DELETE (the common uncontended path skips it entirely), then
                # re-read to see if the slot is free.
                self._cleanup_stale_locks()
                holder = self._lock_held_by(state_id)

            if holder is None:
                self.client.insert(
                    f"{self.schema}.{self.table}_lock",
                    [[state_id, lock_id]],
                    column_names=["state_id", "lock_id"],
                )
                # Verify we won (in case of a concurrent insert race).
                if self._lock_held_by(state_id) == lock_id:
                    acquired = True
                    break
                # Lost the race — withdraw our candidate row and retry.
                self.client.command(
                    f"DELETE FROM {self.lock_table} "  # noqa: S608
                    "WHERE state_id = {state_id:String} AND lock_id = {lock_id:String}",
                    parameters={"state_id": state_id, "lock_id": lock_id},
                )

            time.sleep(retry_seconds)
            seconds_waited += retry_seconds

        if not acquired:
            msg = f"Could not acquire lock for state_id: {state_id}"
            raise StateIDLockedError(msg)

        try:
            yield
        finally:
            self.client.command(
                f"DELETE FROM {self.lock_table} "  # noqa: S608
                "WHERE state_id = {state_id:String} AND lock_id = {lock_id:String}",
                parameters={"state_id": state_id, "lock_id": lock_id},
            )
