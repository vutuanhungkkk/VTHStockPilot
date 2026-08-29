from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_recommendation_is_ranked_and_fully_allocated():
    response = client.post("/api/v1/recommendations", json={"capital": 200_000_000, "risk_level": "balanced",
        "horizon_months": 12, "max_positions": 5, "max_position_weight": .3})
    assert response.status_code == 200
    items = response.json()["recommendations"]
    assert [item["rank"] for item in items] == [1, 2, 3, 4, 5]
    assert abs(sum(item["weight"] for item in items) - 1) < .001
    assert all(item["weight"] <= .3 for item in items)


def test_sector_exclusion_is_enforced():
    response = client.post("/api/v1/recommendations", json={"risk_level": "growth", "max_positions": 4,
        "max_position_weight": .4, "excluded_sectors": ["Technology"]})
    assert response.status_code == 200
    assert all(x["sector"] != "Technology" for x in response.json()["recommendations"])


def test_backtest_has_complete_curve():
    response = client.post("/api/v1/backtests", json={"months": 18, "transaction_cost_bps": 10})
    assert response.status_code == 200
    assert len(response.json()["equity_curve"]) == 18
