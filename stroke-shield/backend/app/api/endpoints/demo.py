"""Demo endpoints.

Prefix: /api/demo

This router returns hard-coded stage payloads for the UI demo mode.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter


router = APIRouter()


STAGES: dict[int, dict[str, Any]] = {
    0: {"stage": 0, "riskScore": 8, "alertTier": "NORMAL", "narrative": "Baseline: all scores low."},
    1: {"stage": 1, "riskScore": 42, "alertTier": "ADVISORY", "narrative": "Cardiac onset: AFib probability rising."},
    2: {"stage": 2, "riskScore": 61, "alertTier": "ADVISORY", "narrative": "Facial asymmetry detected."},
    3: {"stage": 3, "riskScore": 78, "alertTier": "CRITICAL", "narrative": "Speech slurring signals elevated."},
    4: {"stage": 4, "riskScore": 92, "alertTier": "CRITICAL", "narrative": "Multiple streams corroborate high risk."},
}


@router.get("/status")
def status() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/start")
def start_demo() -> dict[str, Any]:
    return STAGES[0]


@router.get("/stage/{stage_number}")
def get_stage(stage_number: int) -> dict[str, Any]:
    stage = int(stage_number)
    return STAGES.get(stage, STAGES[0])


@router.post("/run-full")
def run_full_demo() -> list[dict[str, Any]]:
    return [STAGES[i] for i in sorted(STAGES)]


@router.get("/scenario/{name}")
def get_scenario(name: str) -> dict[str, Any]:
    # Keep this simple for now; frontend can map scenarios.
    return {"name": name, "stage": 0, "message": "Scenario loaded."}


@router.post("/inject-to-fusion")
def inject_to_fusion() -> dict[str, Any]:
    # Placeholder: in a real system we'd call /api/fusion/score.
    return {"ok": True}
