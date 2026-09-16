from typing import List

from pydantic import BaseModel


class CustomerRetentionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str
    risk_factors: List[str]
    recommended_actions: List[str]