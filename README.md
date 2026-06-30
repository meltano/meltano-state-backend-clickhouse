# meltano-state-backend-clickhouse

A [Meltano](https://meltano.com) state backend add-on for **ClickHouse**.

It registers the `clickhouse` state-backend scheme so Meltano can persist
pipeline state (incremental bookmarks, full-table markers, etc.) in a ClickHouse
warehouse — bringing ClickHouse to parity with the Postgres, Snowflake, BigQuery
and MSSQL state backends.

## Install

```sh
pip install "meltano-state-backend-clickhouse @ git+https://github.com/Meltano/meltano-state-backend-clickhouse.git"
```

## Configure

Point Meltano's state backend at a ClickHouse URI:

```sh
export MELTANO_STATE_BACKEND_URI="clickhouse://user:password@host:8123/meltano"
```

or in `meltano.yml`:

```yaml
state_backend:
  uri: clickhouse://user:password@host:8123/meltano
```

Optional per-backend settings (override values parsed from the URI):
`state_backend.clickhouse.{host,port,database,user,password,secure,schema,table}`.

State is stored in `<schema>.<table>` (default `meltano.state`).

## ClickHouse-specific behaviour

ClickHouse is an OLAP store without row-level updates or insert-conflict
constraints, so:

- the **state table** is a `ReplacingMergeTree(updated_at)` keyed by `state_id`;
  reads use `FINAL` so the newest version wins.
- the **lock** is *best-effort* (insert-then-verify with stale-lock cleanup).
  Meltano namespaces state per pipeline via `--state-id-suffix` and the platform
  serialises runs per state id, so contention is minimal in practice.

## License

MIT
