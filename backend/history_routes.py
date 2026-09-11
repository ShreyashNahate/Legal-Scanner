"""
HISTORY / REPOSITORY ROUTES (new file — none of api.py's existing
endpoints or logic are changed).

This module adds a second FastAPI router with the "repository" side of
the project: saving a completed scan, searching past scans, fetching
one in full, deleting one, and a dashboard summary. It reuses the
exact same scan/report/nutrition dicts your existing pipeline already
produces — it does not re-run OCR or re-validate anything.

HOW TO WIRE THIS IN (the one change to api.py):
    from history_routes import router as history_router
    app.include_router(history_router)

That's it — no existing line in api.py needs to move or change.

WHY A SEPARATE /save-scan ENDPOINT INSTEAD OF EDITING /scan-product*:
Keeping saving as an explicit, separate call means every existing
/scan-product* endpoint keeps working exactly as before (nothing about
them changes), and the frontend decides when a scan is worth keeping
(e.g. after the user reviews the result) rather than every scan
attempt automatically becoming permanent history.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import db

router = APIRouter(prefix="", tags=["history"])


@router.on_event("startup")
def _ensure_db():
    db.init_db()


class SaveScanRequest(BaseModel):
    product: dict
    report: dict
    nutrition_analysis: Optional[dict] = None
    scan_method: str = "hybrid"
    image_path: Optional[str] = None


@router.post("/save-scan")
def save_scan(payload: SaveScanRequest):
    """Call this after a /scan-product* call, passing back the exact
    product/report/nutrition_analysis it returned, to persist it."""
    scan_id = db.save_scan(
        product=payload.product,
        report=payload.report,
        nutrition_analysis=payload.nutrition_analysis or {},
        scan_method=payload.scan_method,
        image_path=payload.image_path,
    )
    return {"id": scan_id, "message": "Scan saved."}


@router.get("/scans/{scan_id}")
def get_scan(scan_id: str):
    scan = db.get_scan(scan_id)
    if scan is None:
        raise HTTPException(status_code=404, detail=f"No saved scan with id '{scan_id}'.")
    return scan


@router.delete("/scans/{scan_id}")
def delete_scan(scan_id: str):
    deleted = db.delete_scan(scan_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No saved scan with id '{scan_id}'.")
    return {"message": "Scan deleted."}


@router.get("/scans")
def search_scans(
    q: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Search/list saved scans.
      q          - free text, matches manufacturer name / batch number / MRP
      status     - COMPLIANT | NON_COMPLIANT | NEEDS_REVIEW
      date_from  - ISO date, e.g. 2026-09-01
      date_to    - ISO date, e.g. 2026-09-11
    Returns summaries only; GET /scans/{id} for full product/report detail.
    """
    if status and status not in {"COMPLIANT", "NON_COMPLIANT", "NEEDS_REVIEW"}:
        raise HTTPException(
            status_code=400,
            detail="status must be one of COMPLIANT, NON_COMPLIANT, NEEDS_REVIEW.",
        )
    return db.search_scans(
        query=q, status=status, date_from=date_from, date_to=date_to,
        limit=limit, offset=offset,
    )


@router.get("/dashboard")
def dashboard():
    """Aggregate counts + recent scans for an enforcement dashboard."""
    return db.dashboard_stats()