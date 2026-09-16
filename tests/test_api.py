from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "customer-churn-api"


def test_valid_customer():
    response = client.get("/customer/C03865")

    assert response.status_code == 200

    data = response.json()

    assert data["customer_id"] == "C03865"
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_level"] in ["Low", "Medium", "High"]

    assert isinstance(data["risk_factors"], list)
    assert isinstance(data["recommended_actions"], list)


def test_invalid_customer():
    response = client.get("/customer/C99999")

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Customer C99999 not found"