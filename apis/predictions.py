from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, List, Optional
import logging
import uuid
from datetime import datetime

from app.models.schemas import (
    YieldPredictionRequest, CropRecommendationRequest, DiseasePredictionRequest,
    PestPredictionRequest, RainfallPredictionRequest, SoilTypePredictionRequest,
    PredictionResponse, CropRecommendationResponse, RainfallPredictionResponse,
    SoilTypePredictionResponse, ModelListResponse, AddModelRequest, ModelResponse, PredictionType
)
from app.services.prediction import prediction_service
from app.utils.logging import log_request, log_error, log_ml_prediction
from app.utils.security import get_current_user, api_key_header
from app.database import db_ops
from app.services.ml import ml_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/yield",
    response_model=PredictionResponse,
    summary="Predict crop yield using your Decision Tree model",
    description="Predict crop yield using your loaded crop_yield_model.pkl (Decision Tree Regressor)"
)
async def predict_yield(
    request: YieldPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Predict crop yield using your trained Decision Tree model."""
    log_request(logger, "POST", "/api/predictions/yield", str(current_user["id"]))
    
    try:
        predictions, confidence, recommendations = await prediction_service.predict_yield(request)
        
        # Prepare data for database
        prediction_data = {
            "id": str(uuid.uuid4()),
            "user_id": str(current_user["id"]),
            "prediction_type": PredictionType.YIELD.value,
            "crop_type": request.crop_type.value,
            "input_data": request.dict(),
            "predictions": predictions,
            "confidence": confidence,
            "recommendations": recommendations
        }
        
        # Store prediction in database
        stored_prediction = await db_ops.create_prediction(prediction_data)
        
        # Update user statistics
        current_predictions = current_user.get("predictions_count", 0) + 1
        await db_ops.update_user_stats(
            str(current_user["id"]),
            current_predictions,
            f"{min(95, 70 + current_predictions)}%",
            f"Yield - {request.crop_type.value.title()}"
        )
        
        # Log successful prediction
        log_ml_prediction(logger, "yield", str(current_user["id"]), confidence)
        
        # Create response
        response = PredictionResponse(
            id=uuid.UUID(stored_prediction["id"]),
            user_id=uuid.UUID(stored_prediction["user_id"]),
            prediction_type=stored_prediction["prediction_type"],
            crop_type=stored_prediction["crop_type"],
            predictions=stored_prediction["predictions"],
            confidence=stored_prediction["confidence"],
            recommendations=stored_prediction["recommendations"],
            model_info=predictions.get("model_info", {}),
            created_at=datetime.fromisoformat(stored_prediction["created_at"].replace('Z', '+00:00'))
        )
        
        return response
        
    except Exception as e:
        log_error(logger, e, "Yield prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Yield prediction failed: {str(e)}"
        )

@router.post(
    "/crop-recommendation",
    response_model=CropRecommendationResponse,
    summary="Get crop recommendation using your Random Forest model",
    description="Get optimal crop recommendation using your crop_recommendation_model.pkl (Random Forest with 100 estimators)"
)
async def recommend_crop(
    request: CropRecommendationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get crop recommendation using your trained Random Forest model."""
    log_request(logger, "POST", "/api/predictions/crop-recommendation", str(current_user["id"]))
    
    try:
        predictions, confidence, recommendations = await prediction_service.predict_crop_recommendation(request)
        
        # Prepare data for database
        prediction_data = {
            "id": str(uuid.uuid4()),
            "user_id": str(current_user["id"]),
            "prediction_type": PredictionType.CROP_RECOMMENDATION.value,
            "crop_type": predictions.get("recommended_crop"),
            "input_data": request.dict(),
            "predictions": predictions,
            "confidence": confidence,
            "recommendations": recommendations
        }
        
        # Store prediction in database
        stored_prediction = await db_ops.create_prediction(prediction_data)
        
        # Update user statistics
        current_predictions = current_user.get("predictions_count", 0) + 1
        await db_ops.update_user_stats(
            str(current_user["id"]),
            current_predictions,
            f"{min(95, 70 + current_predictions)}%",
            f"Recommendation - {predictions.get('recommended_crop')}"
        )
        
        # Log successful prediction
        log_ml_prediction(logger, "crop_recommendation", str(current_user["id"]), confidence)
        
        # Create response
        response = CropRecommendationResponse(
            id=uuid.UUID(stored_prediction["id"]),
            user_id=uuid.UUID(stored_prediction["user_id"]),
            prediction_type=stored_prediction["prediction_type"],
            crop_type=stored_prediction["crop_type"],
            predictions=stored_prediction["predictions"],
            confidence=stored_prediction["confidence"],
            recommendations=stored_prediction["recommendations"],
            model_info=predictions.get("model_info", {}),
            created_at=datetime.fromisoformat(stored_prediction["created_at"].replace('Z', '+00:00'))
        )
        
        return response
        
    except Exception as e:
        log_error(logger, e, "Crop recommendation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Crop recommendation failed: {str(e)}"
        )

@router.post(
    "/disease",
    response_model=PredictionResponse,
    summary="Predict crop disease",
    description="Predict crop disease using AI model"
)
async def predict_disease(
    request: DiseasePredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Predict crop disease using AI model."""
    log_request(logger, "POST", "/api/predictions/disease", str(current_user["id"]))
    
    try:
        predictions, confidence, recommendations = await prediction_service.predict_disease(request)
        
        # Prepare data for database
        prediction_data = {
            "id": str(uuid.uuid4()),
            "user_id": str(current_user["id"]),
            "prediction_type": PredictionType.DISEASE.value,
            "crop_type": request.crop_type.value,
            "input_data": request.dict(),
            "predictions": predictions,
            "confidence": confidence,
            "recommendations": recommendations
        }
        
        # Store prediction in database
        stored_prediction = await db_ops.create_prediction(prediction_data)
        
        # Update user statistics
        current_predictions = current_user.get("predictions_count", 0) + 1
        await db_ops.update_user_stats(
            str(current_user["id"]),
            current_predictions,
            f"{min(95, 70 + current_predictions)}%",
            f"Disease - {request.crop_type.value.title()}"
        )
        
        # Log successful prediction
        log_ml_prediction(logger, "disease", str(current_user["id"]), confidence)
        
        # Create response
        response = PredictionResponse(
            id=uuid.UUID(stored_prediction["id"]),
            user_id=uuid.UUID(stored_prediction["user_id"]),
            prediction_type=stored_prediction["prediction_type"],
            crop_type=stored_prediction["crop_type"],
            predictions=stored_prediction["predictions"],
            confidence=stored_prediction["confidence"],
            recommendations=stored_prediction["recommendations"],
            model_info=predictions.get("model_info", {}),
            created_at=datetime.fromisoformat(stored_prediction["created_at"].replace('Z', '+00:00'))
        )
        
        return response
        
    except Exception as e:
        log_error(logger, e, "Disease prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Disease prediction failed: {str(e)}"
        )


@router.post(
    "/pest",
    response_model=PredictionResponse,
    summary="Predict crop pest",
    description="Predict crop pest using AI model"
)
async def predict_pest(
    request: PestPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Predict crop pest using AI model."""
    log_request(logger, "POST", "/api/predictions/pest", str(current_user["id"]))
    
    try:
        predictions, confidence, recommendations = await prediction_service.predict_pest(request)
        
        # Prepare data for database
        prediction_data = {
            "id": str(uuid.uuid4()),
            "user_id": str(current_user["id"]),
            "prediction_type": PredictionType.PEST.value,
            "crop_type": request.crop_type.value,
            "input_data": request.dict(),
            "predictions": predictions,
            "confidence": confidence,
            "recommendations": recommendations
        }
        
        # Store prediction in database
        stored_prediction = await db_ops.create_prediction(prediction_data)
        
        # Update user statistics
        current_predictions = current_user.get("predictions_count", 0) + 1
        await db_ops.update_user_stats(
            str(current_user["id"]),
            current_predictions,
            f"{min(95, 70 + current_predictions)}%",
            f"Pest - {request.crop_type.value.title()}"
        )
        
        # Log successful prediction
        log_ml_prediction(logger, "pest", str(current_user["id"]), confidence)
        
        # Create response
        response = PredictionResponse(
            id=uuid.UUID(stored_prediction["id"]),
            user_id=uuid.UUID(stored_prediction["user_id"]),
            prediction_type=stored_prediction["prediction_type"],
            crop_type=stored_prediction["crop_type"],
            predictions=stored_prediction["predictions"],
            confidence=stored_prediction["confidence"],
            recommendations=stored_prediction["recommendations"],
            model_info=predictions.get("model_info", {}),
            created_at=datetime.fromisoformat(stored_prediction["created_at"].replace('Z', '+00:00'))
        )
        
        return response
        
    except Exception as e:
        log_error(logger, e, "Pest prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pest prediction failed: {str(e)}"
        )


@router.post(
    "/rainfall",
    response_model=RainfallPredictionResponse,
    summary="Predict rainfall using pre-trained model",
    description="Predict rainfall using the loaded rainfall_model.joblib"
)
async def predict_rainfall(
    request: RainfallPredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Predict rainfall using the pre-trained rainfall model."""
    log_request(logger, "POST", "/api/predictions/rainfall", str(current_user["id"]))
    
    try:
        predictions, confidence, recommendations = await prediction_service.predict_rainfall(request)
        
        # Prepare data for database
        prediction_data = {
            "id": str(uuid.uuid4()),
            "user_id": str(current_user["id"]),
            "prediction_type": PredictionType.RAINFALL.value,
            "crop_type": "general",  # Use general for non-crop specific predictions
            "input_data": request.dict(),
            "predictions": predictions,
            "confidence": confidence,
            "recommendations": recommendations
        }
        
        # Store prediction in database
        stored_prediction = await db_ops.create_prediction(prediction_data)
        
        # Update user statistics
        current_predictions = current_user.get("predictions_count", 0) + 1
        await db_ops.update_user_stats(
            str(current_user["id"]),
            current_predictions,
            f"{min(95, 70 + current_predictions)}%",
            f"Rainfall - {predictions.get('predicted_rainfall', 0):.1f}mm"
        )
        
        # Log successful prediction
        log_ml_prediction(logger, "rainfall", str(current_user["id"]), confidence)
        
        # Create response
        response = RainfallPredictionResponse(
            id=uuid.UUID(stored_prediction["id"]),
            user_id=uuid.UUID(stored_prediction["user_id"]),
            prediction_type=stored_prediction["prediction_type"],
            predicted_rainfall=predictions.get("predicted_rainfall", 0.0),
            rainfall_category=predictions.get("rainfall_category", "Unknown"),
            confidence=stored_prediction["confidence"],
            recommendations=stored_prediction["recommendations"],
            model_info=predictions.get("model_info", {}),
            created_at=datetime.fromisoformat(stored_prediction["created_at"].replace('Z', '+00:00'))
        )
        
        return response
        
    except Exception as e:
        log_error(logger, e, "Rainfall prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rainfall prediction failed: {str(e)}"
        )


@router.post(
    "/soil-type",
    response_model=SoilTypePredictionResponse,
    summary="Predict soil type using pre-trained model",
    description="Predict soil type using the loaded soil_model.joblib"
)
async def predict_soil_type(
    request: SoilTypePredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Predict soil type using the pre-trained soil model."""
    log_request(logger, "POST", "/api/predictions/soil-type", str(current_user["id"]))
    
    try:
        predictions, confidence, recommendations = await prediction_service.predict_soil_type(request)
        
        # Prepare data for database
        prediction_data = {
            "id": str(uuid.uuid4()),
            "user_id": str(current_user["id"]),
            "prediction_type": PredictionType.SOIL_TYPE.value,
            "crop_type": "general",  # Use general for non-crop specific predictions
            "input_data": request.dict(),
            "predictions": predictions,
            "confidence": confidence,
            "recommendations": recommendations
        }
        
        # Store prediction in database
        stored_prediction = await db_ops.create_prediction(prediction_data)
        
        # Update user statistics
        current_predictions = current_user.get("predictions_count", 0) + 1
        await db_ops.update_user_stats(
            str(current_user["id"]),
            current_predictions,
            f"{min(95, 70 + current_predictions)}%",
            f"Soil - {predictions.get('predicted_soil_type', 'Unknown')}"
        )
        
        # Log successful prediction
        log_ml_prediction(logger, "soil_type", str(current_user["id"]), confidence)
        
        # Create response
        response = SoilTypePredictionResponse(
            id=uuid.UUID(stored_prediction["id"]),
            user_id=uuid.UUID(stored_prediction["user_id"]),
            prediction_type=stored_prediction["prediction_type"],
            predicted_soil_type=predictions.get("predicted_soil_type", "Unknown"),
            confidence=stored_prediction["confidence"],
            alternative_soil_types=predictions.get("alternative_soil_types", []),
            soil_probabilities=predictions.get("soil_probabilities", {}),
            recommendations=stored_prediction["recommendations"],
            model_info=predictions.get("model_info", {}),
            created_at=datetime.fromisoformat(stored_prediction["created_at"].replace('Z', '+00:00'))
        )
        
        return response
        
    except Exception as e:
        log_error(logger, e, "Soil type prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Soil type prediction failed: {str(e)}"
        )


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="List available ML models",
    description="Get list of all loaded ML models and their configurations"
)
async def list_models(
    current_user: dict = Depends(get_current_user)
):
    """List all available ML models."""
    log_request(logger, "GET", "/api/predictions/models", str(current_user["id"]))
    
    try:
        available_models = await ml_service.get_available_models()
        return ModelListResponse(models=available_models)
    except Exception as e:
        log_error(logger, e, "List models")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list models"
        )


@router.post(
    "/add-model",
    response_model=ModelResponse,
    summary="Add new ML model",
    description="Upload and add new .pkl model to the system"
)
async def add_model(
    request: AddModelRequest,
    current_user: dict = Depends(get_current_user)
):
    """Add a new ML model to the system."""
    log_request(logger, "POST", "/api/predictions/add-model", str(current_user["id"]))
    
    try:
        # Check if user is admin (implement proper auth later)
        if current_user.get("role") != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required"
            )
        
        # Add the model
        success = await ml_service.add_new_model(request.model_name, request.model_path)
        
        if not success:
            raise Exception("Failed to add model")
        
        return ModelResponse(
            model_name=request.model_name,
            loaded=True,
            message="Model added successfully"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        log_error(logger, e, "Add model")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add model: {str(e)}"
        )


@router.get(
    "/history",
    response_model=List[PredictionResponse],
    summary="Get prediction history",
    description="Get user's prediction history"
)
async def get_prediction_history(
    limit: int = Query(10, ge=1, le=50, description="Number of predictions to return"),
    offset: int = Query(0, ge=0, description="Number of predictions to skip"),
    prediction_type: Optional[PredictionType] = Query(None, description="Filter by prediction type"),
    current_user: dict = Depends(get_current_user)
):
    """Get user's prediction history."""
    log_request(logger, "GET", "/api/predictions/history", str(current_user["id"]))
    
    try:
        predictions = await db_ops.get_user_predictions(
            str(current_user["id"]),
            limit=limit,
            offset=offset,
            prediction_type=prediction_type.value if prediction_type else None
        )
        
        # Convert to response models
        response = []
        for pred in predictions:
            response.append(PredictionResponse(
                id=uuid.UUID(pred["id"]),
                user_id=uuid.UUID(pred["user_id"]),
                prediction_type=pred["prediction_type"],
                crop_type=pred["crop_type"],
                predictions=pred["predictions"],
                confidence=pred["confidence"],
                recommendations=pred["recommendations"],
                model_info=pred.get("model_info", {}),
                created_at=datetime.fromisoformat(pred["created_at"].replace('Z', '+00:00'))
            ))
        
        return response
        
    except Exception as e:
        log_error(logger, e, "Prediction history")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get prediction history"
        )