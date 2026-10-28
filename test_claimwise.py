"""Automated tests for the ClaimWise API, tools and guardrails."""

from fastapi.testclient import TestClient

from api import app
from rag import answer_question
from tools import predict_risk

client = TestClient(app)

HIGH_RISK = {
    "age": 45, "sex": "male", "bmi": 31, "children": 2,
    "smoker": "yes", "region": "southeast",
}
LOW_RISK = {
    "age": 30, "sex": "female", "bmi": 24, "children": 1,
    "smoker": "no", "region": "northwest",
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_rejects_underage_customer():
    bad_customer = {**HIGH_RISK, "age": 15}
    response = client.post("/predict", json=bad_customer)
    assert response.status_code == 422


def test_tool_flags_high_risk_customer():
    result = predict_risk.invoke(HIGH_RISK)
    assert "HIGH cost" in result


def test_tool_low_risk_customer():
    result = predict_risk.invoke(LOW_RISK)
    assert "not high cost" in result


def test_tool_returns_error_for_bad_input():
    result = predict_risk.invoke({**HIGH_RISK, "smoker": "maybe"})
    assert "Invalid customer details" in result


def test_off_topic_question_skips_llm():
    result = answer_question("What is the capital of France?")
    assert result["used_llm"] is False
    assert "I don't know" in result["answer"]