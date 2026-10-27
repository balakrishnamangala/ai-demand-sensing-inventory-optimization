"""Pydantic request/response schemas for the API."""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


class ForecastRequest(BaseModel):
    sku_ids: List[str] = Field(..., description="SKU identifiers to forecast")
    horizon_weeks: int = Field(default=12, ge=1, le=52)
    include_components: bool = Field(default=False)


class ForecastItem(BaseModel):
    sku_id: str
    date: date
    forecast: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    prophet_component: Optional[float] = None
    xgboost_component: Optional[float] = None


class ForecastResponse(BaseModel):
    forecasts: List[ForecastItem]
    model: str = "ensemble"
    generated_at: str


class InventoryRequest(BaseModel):
    sku_ids: List[str]
    service_level: float = Field(default=0.98, ge=0.80, le=0.9999)


class InventoryItem(BaseModel):
    sku_id: str
    safety_stock: float
    reorder_point: float
    eoq: Optional[float] = None
    abc_class: str
    xyz_class: str
    segment: str
    policy: str


class InventoryResponse(BaseModel):
    inventory_params: List[InventoryItem]
    total_working_capital: float
    avg_service_level: float


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
