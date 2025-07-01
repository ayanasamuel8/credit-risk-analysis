from pydantic import BaseModel


class Transaction(BaseModel):
    TransactionId: str
    CustomerId: str
    ProductCategory: str
    ChannelId: str
    Amount: float
    Value: float
    TransactionStartTime: str
    PricingStrategy: str
    SubscriptionId: str
    ProviderId: str
    ProductId: str


class PredictionResponse(BaseModel):
    is_high_risk: int
