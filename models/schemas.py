"""
Pydantic models and schemas for AgriSmart API.
Fixed for Pydantic v2 compatibility.
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import re
from uuid import UUID

class PredictionType(str, Enum):
    """Supported prediction types."""
    YIELD = "yield"
    DISEASE = "disease"
    PEST = "pest"
    CROP_RECOMMENDATION = "crop_recommendation"
    RAINFALL = "rainfall"
    SOIL_TYPE = "soil_type"

class CropType(str, Enum):
    """Supported crop types."""
    WHEAT = "wheat"
    RICE = "rice"
    CORN = "corn"
    COTTON = "cotton"
    SUGARCANE = "sugarcane"
    TOMATO = "tomato"
    POTATO = "potato"
    ONION = "onion"
    MAIZE = "maize"
    BARLEY = "barley"
    SOYBEAN = "soybean"
    CHICKPEA = "chickpea"
    LENTIL = "lentil"
    GROUNDNUT = "groundnut"

# User Models
class UserCreate(BaseModel):
    """User registration model."""
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    phone: str = Field(..., min_length=13, max_length=15)
    region: str = Field(..., max_length=100)
    farm_size: float = Field(..., ge=0)
    main_crops: str = Field(..., max_length=200)
    password: str = Field(..., min_length=12)
    
    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v):
        pattern = r"^\+\d{2}-\d{10}$"
        if not re.match(pattern, v):
            raise ValueError('Phone must be in format +XX-XXXXXXXXXX')
        return v
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        pattern = r"^[A-Za-z\d@$!%*#?&]{12,}$"
        if not re.match(pattern, v):
            raise ValueError('Password must be at least 12 characters with letters, numbers, and special characters')
        return v

class UserLogin(BaseModel):
    """User login model."""
    email: EmailStr
    password: str

class GoogleAuthRequest(BaseModel):
    """Google OAuth request model."""
    token: str

class UserResponse(BaseModel):
    """User response model."""
    id: UUID
    name: str
    email: str
    phone: str
    region: str
    farm_size: float
    main_crops: str
    member_since: int
    predictions_count: int = 0
    accuracy_rate: str = "0%"
    last_prediction: str = "Never"
    created_at: datetime

class TokenResponse(BaseModel):
    """JWT token response model."""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

# Prediction Models
class PredictionRequest(BaseModel):
    """Base prediction request model."""
    prediction_type: PredictionType
    crop_type: Optional[CropType] = None
    area: float = Field(..., ge=0)
    soil_data: Dict[str, Any] = {}
    weather_data: Dict[str, Any] = {}
    additional_params: Optional[Dict[str, Any]] = {}

class YieldPredictionRequest(PredictionRequest):
    """Enhanced yield prediction request matching your crop_yield_model.pkl."""
    prediction_type: PredictionType = PredictionType.YIELD
    crop_type: CropType
    
    # Core parameters for your Decision Tree model (6 features)
    rainfall: float = Field(..., ge=0, description="Rain Fall (mm)")
    fertilizer_amount: float = Field(50.0, ge=0, description="Fertilizer amount")
    temperature: float = Field(..., ge=-50, le=60, description="Temperature")
    nitrogen: float = Field(..., ge=0, description="Nitrogen (N)")
    phosphorus: float = Field(..., ge=0, description="Phosphorus (P)")  
    potassium: float = Field(..., ge=0, description="Potassium (K)")
    
    # Additional parameters for API compatibility
    sowing_date: str
    soil_ph: float = Field(..., ge=0, le=14)
    
    @field_validator('sowing_date')
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

class CropRecommendationRequest(BaseModel):
    """New model for your crop_recommendation_model.pkl (Random Forest with 11 features)."""
    prediction_type: PredictionType = PredictionType.CROP_RECOMMENDATION
    area: float = Field(..., ge=0)
    
    # 11 features required by your Random Forest model
    nitrogen: float = Field(..., ge=0, description="Nitrogen (N)")
    phosphorus: float = Field(..., ge=0, description="Phosphorus (P)")
    potassium: float = Field(..., ge=0, description="Potassium (K)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    ec: float = Field(..., ge=0, description="Electrical Conductivity")
    sulfur: float = Field(..., ge=0, description="Sulfur (S)")
    copper: float = Field(..., ge=0, description="Copper (Cu)")
    iron: float = Field(..., ge=0, description="Iron (Fe)")
    manganese: float = Field(..., ge=0, description="Manganese (Mn)")
    zinc: float = Field(..., ge=0, description="Zinc (Zn)")
    boron: float = Field(..., ge=0, description="Boron (B)")
    
    # Optional metadata
    region: Optional[str] = None
    season: Optional[str] = None
    climate_data: Optional[Dict[str, Any]] = {}

class DiseasePredictionRequest(PredictionRequest):
    """Disease prediction specific request."""
    prediction_type: PredictionType = PredictionType.DISEASE
    crop_type: CropType
    symptoms: List[str] = []
    affected_area_percentage: float = Field(..., ge=0, le=100)
    days_since_symptoms: int = Field(..., ge=0)

class PestPredictionRequest(PredictionRequest):
    """Pest prediction specific request."""
    prediction_type: PredictionType = PredictionType.PEST
    crop_type: CropType
    pest_description: str
    damage_level: str = Field(..., pattern=r"^(low|medium|high)$")  # FIXED: Changed regex to pattern
    treatment_history: Optional[List[str]] = []

class RainfallPredictionRequest(BaseModel):
    """Rainfall prediction request model."""
    prediction_type: PredictionType = PredictionType.RAINFALL
    year: int = Field(2024, ge=2000, le=2030, description="Year")
    subdivision: int = Field(1, ge=1, le=10, description="Subdivision/Region code")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    current_rainfall: float = Field(0.0, ge=0, description="Current month rainfall in mm")
    
    # Optional metadata
    location: Optional[str] = None
    elevation: Optional[float] = None

class SoilTypePredictionRequest(BaseModel):
    """Soil type prediction request model."""
    prediction_type: PredictionType = PredictionType.SOIL_TYPE
    nitrogen: float = Field(..., ge=0, description="Nitrogen content")
    phosphorus: float = Field(..., ge=0, description="Phosphorus content")
    potassium: float = Field(..., ge=0, description="Potassium content")
    temperature: float = Field(..., ge=-50, le=60, description="Temperature in Celsius")
    moisture: float = Field(..., ge=0, le=100, description="Moisture percentage")
    humidity: float = Field(..., ge=0, le=100, description="Humidity percentage")
    
    # Optional metadata
    location: Optional[str] = None
    depth: Optional[float] = Field(None, ge=0, description="Soil depth in cm")
    texture: Optional[str] = None

class PredictionResponse(BaseModel):
    """Enhanced prediction response model."""
    id: UUID
    user_id: UUID
    prediction_type: str
    crop_type: Optional[str] = None
    predictions: Dict[str, Any]
    confidence: float
    recommendations: Dict[str, Any]
    model_info: Optional[Dict[str, Any]] = {}
    created_at: datetime
    
    model_config = {"protected_namespaces": ()}  # FIX PYDANTIC WARNING

class CropRecommendationResponse(BaseModel):
    """Specific response for crop recommendation."""
    id: UUID
    user_id: UUID
    recommended_crop: str
    confidence: float
    alternative_crops: List[str]
    crop_probabilities: Optional[Dict[str, float]] = {}
    recommendations: Dict[str, Any]
    soil_analysis: Dict[str, Any]
    model_info: Dict[str, Any]
    created_at: datetime
    
    model_config = {"protected_namespaces": ()}  # FIX PYDANTIC WARNING

class RainfallPredictionResponse(BaseModel):
    """Specific response for rainfall prediction."""
    id: UUID
    user_id: UUID
    prediction_type: str
    predicted_rainfall: float
    rainfall_category: str
    confidence: float
    recommendations: Dict[str, Any]
    model_info: Dict[str, Any]
    created_at: datetime
    
    model_config = {"protected_namespaces": ()}  # FIX PYDANTIC WARNING

class SoilTypePredictionResponse(BaseModel):
    """Specific response for soil type prediction."""
    id: UUID
    user_id: UUID
    prediction_type: str
    predicted_soil_type: str
    confidence: float
    alternative_soil_types: List[str]
    soil_probabilities: Optional[Dict[str, float]] = {}
    recommendations: Dict[str, Any]
    model_info: Dict[str, Any]
    created_at: datetime
    
    model_config = {"protected_namespaces": ()}  # FIX PYDANTIC WARNING

# Model Management
class ModelInfo(BaseModel):
    """Information about a loaded ML model."""
    name: str
    type: str
    algorithm: str
    input_features: List[str]
    feature_count: int
    output_type: str
    loaded: bool
    is_fallback: bool = False
    file_path: Optional[str] = None
    accuracy: Optional[float] = None
    last_updated: Optional[datetime] = None

class ModelListResponse(BaseModel):
    """Response for listing available models."""
    models: Dict[str, ModelInfo]
    total_models: int
    loaded_models: int
    fallback_models: int

class AddModelRequest(BaseModel):
    """Request to add a new model."""
    model_name: str = Field(..., min_length=1, max_length=100)
    model_path: str
    model_type: str = Field(..., pattern=r"^(regressor|classifier|custom)$")
    description: Optional[str] = None
    expected_features: Optional[List[str]] = []
    
    model_config = {"protected_namespaces": ()}  # FIX PYDANTIC WARNING

class ModelResponse(BaseModel):
    """Model operation response."""
    model_name: str
    loaded: bool
    message: str
    
    model_config = {"protected_namespaces": ()}  # FIX PYDANTIC WARNING

# Irrigation Models
class IrrigationRequest(BaseModel):
    """Irrigation schedule request model."""
    crop_type: CropType
    area: float = Field(..., ge=0)
    soil_moisture: float = Field(..., ge=0, le=100)
    rainfall: float = Field(..., ge=0)
    temperature: float = Field(..., ge=-50, le=60)
    last_irrigation: str
    
    # Additional parameters for better scheduling
    growth_stage: Optional[str] = Field(None, pattern=r"^(seedling|vegetative|flowering|fruiting|maturity)$")  # FIXED
    irrigation_method: Optional[str] = Field(None, pattern=r"^(drip|sprinkler|flood|furrow)$")  # FIXED
    
    @field_validator('last_irrigation')
    @classmethod
    def validate_last_irrigation(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

class IrrigationResponse(BaseModel):
    """Irrigation schedule response model."""
    schedule_date: datetime
    duration_minutes: int
    water_volume: float
    recommendations: Dict[str, Any]
    next_irrigation: Optional[datetime] = None
    efficiency_score: Optional[float] = None

# Dashboard Models
class DashboardStats(BaseModel):
    """Enhanced dashboard statistics model."""
    total_predictions: int = 0
    accuracy_rate: str = "0%"
    last_prediction: str = "Never"
    irrigation_count: int = 0
    member_since: int = 2025
    recent_predictions: List[PredictionResponse] = []
    
    # Enhanced stats
    models_available: int = 0
    crop_recommendations: int = 0
    yield_predictions: int = 0
    disease_detections: int = 0
    pest_classifications: int = 0

class CropAnalytics(BaseModel):
    """Enhanced crop analytics model."""
    crop_type: str
    total_area: float
    avg_yield: float
    disease_incidents: int
    pest_incidents: int
    irrigation_frequency: float
    
    # New analytics
    recommended_frequency: int = 0
    success_rate: float = 0.0
    soil_suitability: Optional[str] = None

# Soil and Weather Data Models
class EnhancedSoilData(BaseModel):
    """Enhanced soil data model matching your crop recommendation model."""
    # Primary nutrients
    nitrogen: float = Field(..., ge=0, description="Nitrogen (N)")
    phosphorus: float = Field(..., ge=0, description="Phosphorus (P)")
    potassium: float = Field(..., ge=0, description="Potassium (K)")
    
    # Soil properties
    ph: float = Field(..., ge=0, le=14, description="Soil pH")
    ec: float = Field(..., ge=0, description="Electrical Conductivity")
    organic_matter: Optional[float] = Field(None, ge=0)
    moisture: Optional[float] = Field(None, ge=0, le=100)
    
    # Micronutrients (for your model)
    sulfur: float = Field(..., ge=0, description="Sulfur (S)")
    copper: float = Field(..., ge=0, description="Copper (Cu)")
    iron: float = Field(..., ge=0, description="Iron (Fe)")
    manganese: float = Field(..., ge=0, description="Manganese (Mn)")
    zinc: float = Field(..., ge=0, description="Zinc (Zn)")
    boron: float = Field(..., ge=0, description="Boron (B)")
    
    # Additional properties
    texture: Optional[str] = None
    drainage: Optional[str] = None
    depth: Optional[float] = None

class WeatherData(BaseModel):
    """Weather data model."""
    temperature: float
    humidity: float
    rainfall: float
    wind_speed: float
    pressure: float
    date: datetime
    
    # Additional weather parameters
    uv_index: Optional[float] = None
    evapotranspiration: Optional[float] = None

# Error Response Models
class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    message: str
    status_code: int

class ValidationErrorResponse(BaseModel):
    """Validation error response model."""
    error: str = "validation_error"
    details: List[Dict[str, Any]]
    status_code: int = 422

# Success Response Models
class SuccessResponse(BaseModel):
    """Generic success response model."""
    message: str
    status_code: int = 200
    data: Optional[Dict[str, Any]] = None