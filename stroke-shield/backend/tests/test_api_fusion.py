from __future__ import annotations


def test_fusion_status(client):
    resp = client.get("/api/fusion/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_fusion_score_and_history_and_reset(client):
    # Score a fusion result.
    resp = client.post("/api/fusion/score", json={"scores": {"afib_cnn": 0.8, "cardiac_if": 0.2}})
    assert resp.status_code == 200
    payload = resp.json()
    assert "risk_score" in payload
    assert 0.0 <= payload["risk_score"] <= 100.0

    # History should contain at least one entry now.
    hist = client.get("/api/fusion/history")
    assert hist.status_code == 200
    assert isinstance(hist.json(), list)
    assert len(hist.json()) >= 1

    # Reset clears history.
    reset = client.post("/api/fusion/reset")
    assert reset.status_code == 200
    assert reset.json().get("cleared") is True
