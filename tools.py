"""Tools the ClaimWise agent can use: the ML risk model and the policy search."""

from langchain_core.tools import tool
from pydantic import ValidationError

from api import Customer, model, to_features
from rag import answer_question


@tool
def predict_risk(age: int, sex: str, bmi: float, children: int, smoker: str, region: str) -> str:
    """Predict whether an insurance customer is likely to be high-cost, using the trained ML model.

    Use this when the user gives customer details and asks about their risk or cost.

    Args:
        age: Age in years, 18 to 100.
        sex: "female" or "male".
        bmi: Body mass index, for example 27.5.
        children: Number of children, 0 to 10.
        smoker: "yes" or "no".
        region: One of "northeast", "northwest", "southeast", "southwest".
    """
    try:
        customer = Customer(
            age=age, sex=sex, bmi=bmi, children=children, smoker=smoker, region=region
        )
    except ValidationError as error:
        problem = error.errors()[0]
        return f"Invalid customer details: {problem['loc'][0]} - {problem['msg']}"

    features = to_features(customer)
    probability = round(float(model.predict_proba(features)[0][1]), 3)
    label = "HIGH cost" if probability >= 0.5 else "not high cost"
    return f"The model predicts this customer is {label} (probability {probability})."


@tool
def ask_policy(question: str) -> str:
    """Answer a question about the health insurance policy document.

    Use this for questions about deductibles, copays, coinsurance, coverage, or limits.
    The answer includes page citations.

    Args:
        question: The policy question in plain English.
    """
    result = answer_question(question)
    return result["answer"]


if __name__ == "__main__":
    print("--- Testing predict_risk ---")
    print(predict_risk.invoke({
        "age": 45, "sex": "male", "bmi": 31, "children": 2,
        "smoker": "yes", "region": "southeast",
    }))
    print(predict_risk.invoke({
        "age": 15, "sex": "male", "bmi": 31, "children": 2,
        "smoker": "yes", "region": "southeast",
    }))

    print("\n--- Testing ask_policy ---")
    print(ask_policy.invoke({"question": "What is the deductible?"}))

    print("\n--- What the AI will see ---")
    print("Name:", predict_risk.name)
    print("Inputs:", list(predict_risk.args.keys()))