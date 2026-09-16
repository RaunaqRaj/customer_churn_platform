from fastapi import FastAPI, HTTPException

from src.models.retention_engine import (
    load_training_data,
    get_test_customer_mapping,
    load_model,
    create_explainer,
    analyze_customer,
)

from src.api.schemas import CustomerRetentionResponse


app = FastAPI(
    title="Customer Churn Prediction & Retention API",
    description="API for customer churn prediction, explainability, and retention recommendations.",
    version="1.0.0",
)


# Load model and data once when the API starts
_, _, X_test = load_training_data()
customer_ids = get_test_customer_mapping()
model = load_model()
explainer = create_explainer(model)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "customer-churn-api"
    }


@app.get(
    "/customer/{customer_id}",
    response_model=CustomerRetentionResponse
)
def get_customer_retention(customer_id: str):

    if customer_id not in customer_ids.values:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found"
        )

    result = analyze_customer(
        customer_id,
        model,
        explainer,
        X_test,
        customer_ids
    )

    return result