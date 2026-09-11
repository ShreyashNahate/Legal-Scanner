"""
REPOSITORY MODULE (new — does not modify any existing pipeline file).

Purpose: persist every completed scan (product fields + compliance
report + nutrition analysis + a copy of the source image) so scans
survive past a single request/response, and so they can later be
searched, listed on a dashboard, or exported as a report.

Storage: SQLite file (dataset/scans.db). No extra infra, no cost —
matches the rest of this project's zero-cost constraint. If this
later needs to move to Postgres/MySQL, only this file changes; nothing
that calls save_scan()/get_scan()/search_scans() needs to know.

Design notes:
  - product / report / nutrition_analysis are stored as JSON TEXT
    columns (they're nested dicts already produced by the existing
    pipeline) rather than being exploded into many SQL columns. This
    keeps the schema stable even as fields get added upstream.
  - A handful of the most-searched product fields are ALSO copied out
    into plain columns (manufacturer_name, mrp, status, etc.) purely
    so search_scans() can filter/sort with normal SQL instead of
    parsing JSON on every row.
  - image_path stores where the uploaded image was saved by
    api.py's _save_upload() (input_product/api_uploads/<filename>) —
    this module does not copy or move the image, it just records the
    path so a scan record can always be traced back to its source
    photo ("attachment of photographs and supporting evidence").
"""

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = "dataset/scans.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id                  TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    scan_method         TEXT,
    image_path          TEXT,

    manufacturer_name   TEXT,
    net_quantity        TEXT,
    mrp                 TEXT,
    batch_number        TEXT,
    manufacturing_date  TEXT,
    expiry_date         TEXT,

    checks_passed       INTEGER,
    checks_failed       INTEGER,
    checks_review       INTEGER,
    checks_total        INTEGER,
    overall_status       TEXT,   -- COMPLIANT / NON_COMPLIANT / NEEDS_REVIEW

    product_json         TEXT NOT NULL,
    report_json           TEXT NOT NULL,
    nutrition_json        TEXT
);

CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at);
CREATE INDEX IF NOT EXISTS idx_scans_manufacturer ON scans(manufacturer_name);
CREATE INDEX IF NOT EXISTS idx_scans_status ON scans(overall_status);
"""


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Safe to call on every startup — CREATE TABLE IF NOT EXISTS is a no-op
    once the table exists."""
    with _connect() as conn:
        conn.executescript(SCHEMA)


def _overall_status(summary: dict) -> str:
    """Simple, transparent rollup from the existing report summary counts.
    A human always reviews the underlying results — this is just a label
    for sorting/filtering, not a new compliance decision."""
    if (summary.get("failed") or 0) > 0:
        return "NON_COMPLIANT"
    if (summary.get("review") or 0) > 0:
        return "NEEDS_REVIEW"
    return "COMPLIANT"


def save_scan(product: dict, report: dict, nutrition_analysis: dict,
              scan_method: str, image_path: str | None = None) -> str:
    """Persists one completed scan. Returns the generated scan id.

    Takes the exact dicts already produced by the existing pipeline
    (build_product_json / run_rule_engine / analyze_nutrition_and_claims
    output shapes) — no reshaping of their logic, just storage."""
    scan_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    summary = (report or {}).get("summary", {})

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO scans (
                id, created_at, scan_method, image_path,
                manufacturer_name, net_quantity, mrp, batch_number,
                manufacturing_date, expiry_date,
                checks_passed, checks_failed, checks_review, checks_total,
                overall_status,
                product_json, report_json, nutrition_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scan_id, created_at, scan_method, image_path,
                product.get("manufacturer_name", ""),
                product.get("net_quantity", ""),
                product.get("mrp", ""),
                product.get("batch_number", ""),
                product.get("manufacturing_date", ""),
                product.get("expiry_date", ""),
                summary.get("passed", 0),
                summary.get("failed", 0),
                summary.get("review", 0),
                summary.get("total_checks", 0),
                _overall_status(summary),
                json.dumps(product, ensure_ascii=False),
                json.dumps(report, ensure_ascii=False),
                json.dumps(nutrition_analysis or {}, ensure_ascii=False),
            ),
        )
    return scan_id


def _row_to_dict(row: sqlite3.Row, include_full_json: bool) -> dict:
    result = {
        "id": row["id"],
        "created_at": row["created_at"],
        "scan_method": row["scan_method"],
        "image_path": row["image_path"],
        "manufacturer_name": row["manufacturer_name"],
        "net_quantity": row["net_quantity"],
        "mrp": row["mrp"],
        "batch_number": row["batch_number"],
        "manufacturing_date": row["manufacturing_date"],
        "expiry_date": row["expiry_date"],
        "summary": {
            "passed": row["checks_passed"],
            "failed": row["checks_failed"],
            "review": row["checks_review"],
            "total_checks": row["checks_total"],
        },
        "overall_status": row["overall_status"],
    }
    if include_full_json:
        result["product"] = json.loads(row["product_json"])
        result["report"] = json.loads(row["report_json"])
        result["nutrition_analysis"] = (
            json.loads(row["nutrition_json"]) if row["nutrition_json"] else {}
        )
    return result


def get_scan(scan_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    return _row_to_dict(row, include_full_json=True) if row else None


def delete_scan(scan_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    return cur.rowcount > 0


def search_scans(
    query: str | None = None,
    status: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    """
    query    - free-text match against manufacturer_name/batch_number/mrp
    status   - COMPLIANT / NON_COMPLIANT / NEEDS_REVIEW
    date_from/date_to - ISO date strings, filters on created_at
    Returns {"total": N, "results": [...]} (list rows are summaries,
    not full product/report JSON — call get_scan(id) for full detail).
    """
    clauses = []
    params: list = []

    if query:
        clauses.append(
            "(manufacturer_name LIKE ? OR batch_number LIKE ? OR mrp LIKE ?)"
        )
        like = f"%{query}%"
        params.extend([like, like, like])

    if status:
        clauses.append("overall_status = ?")
        params.append(status)

    if date_from:
        clauses.append("created_at >= ?")
        params.append(date_from)

    if date_to:
        clauses.append("created_at <= ?")
        params.append(date_to)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with _connect() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) AS c FROM scans {where_sql}", params
        ).fetchone()["c"]

        rows = conn.execute(
            f"""
            SELECT * FROM scans {where_sql}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()

    return {
        "total": total,
        "results": [_row_to_dict(r, include_full_json=False) for r in rows],
    }


def dashboard_stats() -> dict:
    """Aggregate counts for the enforcement dashboard — total scans,
    breakdown by status, and the most recent scans."""
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM scans").fetchone()["c"]
        by_status_rows = conn.execute(
            "SELECT overall_status, COUNT(*) AS c FROM scans GROUP BY overall_status"
        ).fetchall()
        recent_rows = conn.execute(
            "SELECT * FROM scans ORDER BY created_at DESC LIMIT 10"
        ).fetchall()

    return {
        "total_scans": total,
        "by_status": {r["overall_status"]: r["c"] for r in by_status_rows},
        "recent_scans": [_row_to_dict(r, include_full_json=False) for r in recent_rows],
    }