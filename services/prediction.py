import logging
from typing import Dict, Any, List, Tuple
import uuid
from datetime import datetime

from app.models.schemas import (
    YieldPredictionRequest, CropRecommendationRequest, DiseasePredictionRequest, 
    PestPredictionRequest, RainfallPredictionRequest, SoilTypePredictionRequest,
    PredictionResponse, CropRecommendationResponse, RainfallPredictionResponse,
    SoilTypePredictionResponse
)
from app.services.ml import ml_service
from app.database import db_ops
from app.utils.logging import log_ml_prediction, log_error

logger = logging.getLogger(__name__)

class PredictionService:
    """Prediction service for handling ML predictions and database operations."""
    
    def __init__(self):
        self.ml_service = ml_service
        self.db_ops = db_ops
    
    async def predict_yield(self, request: YieldPredictionRequest) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Predict crop yield using the Decision Tree model."""
        try:
            # Prepare input for your crop_yield_model.pkl (6 features)
            input_features = {
                'Rain Fall (mm)': request.rainfall,
                'Fertilizer': request.fertilizer_amount,
                'Temperatue': request.temperature,  # Note: keeping the typo from your description
                'Nitrogen (N)': request.nitrogen,
                'Phosphorus (P)': request.phosphorus,
                'Potassium (K)': request.potassium
            }
            
            # Get prediction from ML service
            predictions = await self.ml_service.predict_yield(input_features)
            confidence = predictions.get('confidence', 0.95)
            
            # Generate recommendations based on prediction
            recommendations = self._generate_yield_recommendations(predictions, request)
            
            logger.info(f"Yield prediction completed for crop {request.crop_type.value}")
            return predictions, confidence, recommendations
            
        except Exception as e:
            logger.error(f"Yield prediction error: {str(e)}")
            raise
    
    async def predict_crop_recommendation(self, request: CropRecommendationRequest) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Get crop recommendation using the Random Forest model."""
        try:
            # Prepare input for your crop_recommendation_model.pkl (11 features)
            input_features = {
                'N': request.nitrogen,
                'P': request.phosphorus,
                'K': request.potassium,
                'temperature': request.temperature,
                'humidity': request.humidity,
                'ph': request.ph,
                'rainfall': request.rainfall,
                'EC': request.ec,
                'S': request.sulfur,
                'Cu': request.copper,
                'Fe': request.iron,
                'Mn': request.manganese,
                'Zn': request.zinc,
                'B': request.boron
            }
            
            # Get prediction from ML service
            predictions = await self.ml_service.predict_crop_recommendation(input_features)
            confidence = predictions.get('confidence', 0.95)
            
            # Generate recommendations based on prediction
            recommendations = self._generate_crop_recommendations(predictions, request)
            
            logger.info(f"Crop recommendation completed: {predictions.get('recommended_crop', 'Unknown')}")
            return predictions, confidence, recommendations
            
        except Exception as e:
            logger.error(f"Crop recommendation error: {str(e)}")
            raise
    
    async def predict_disease(self, request: DiseasePredictionRequest) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Predict crop disease."""
        try:
            # Get prediction from ML service
            predictions = await self.ml_service.predict_disease(request)
            confidence = predictions.get('confidence', 0.9)
            
            # Generate recommendations
            recommendations = self._generate_disease_recommendations(predictions, request)
            
            logger.info(f"Disease prediction completed for crop {request.crop_type.value}")
            return predictions, confidence, recommendations
            
        except Exception as e:
            logger.error(f"Disease prediction error: {str(e)}")
            raise
    
    async def predict_pest(self, request: PestPredictionRequest) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Predict crop pest."""
        try:
            # Get prediction from ML service
            predictions, confidence, recommendations = await self.ml_service.predict_pest(request)
            
            logger.info(f"Pest prediction completed for crop {request.crop_type.value}")
            return predictions, confidence, recommendations
            
        except Exception as e:
            logger.error(f"Pest prediction error: {str(e)}")
            raise
    
    def _generate_yield_recommendations(self, predictions: Dict[str, Any], request: YieldPredictionRequest) -> Dict[str, Any]:
        """Generate yield-specific recommendations."""
        recommendations = {
            "fertilizer": [],
            "irrigation": [],
            "general": []
        }
        
        predicted_yield = predictions.get('yield', 0.0)
        
        if request.nitrogen < 30:
            recommendations["fertilizer"].append("Increase nitrogen application")
        if request.phosphorus < 20:
            recommendations["fertilizer"].append("Apply phosphorus fertilizer")
        if request.potassium < 30:
            recommendations["fertilizer"].append("Apply potassium fertilizer")
        
        if request.soil_ph < 6.0:
            recommendations["general"].append("Consider lime application to raise soil pH")
        elif request.soil_ph > 7.5:
            recommendations["general"].append("Consider sulfur application to lower soil pH")
        
        if predicted_yield < 2000:
            recommendations["irrigation"].append("Optimize irrigation schedule")
        
        return recommendations
    
    def _generate_crop_recommendations(self, predictions: Dict[str, Any], request: CropRecommendationRequest) -> Dict[str, Any]:
        """Generate crop recommendation-specific recommendations."""
        recommended_crop = predictions.get('recommended_crop', 'wheat')
        
        recommendations = {
            "planting": [f"Plant {recommended_crop} in the recommended season"],
            "soil_preparation": [],
            "maintenance": []
        }
        
        if request.ph < 6.0:
            recommendations["soil_preparation"].append("Lime application recommended")
        elif request.ph > 8.0:
            recommendations["soil_preparation"].append("Sulfur application recommended")
        
        recommendations["maintenance"].extend([
            "Regular soil testing recommended",
            "Follow recommended fertilizer schedule",
            "Monitor weather conditions closely"
        ])
        
        return recommendations
    
    def _generate_disease_recommendations(self, predictions: Dict[str, Any], request: DiseasePredictionRequest) -> Dict[str, Any]:
        """Generate disease-specific recommendations."""
        disease_risk = predictions.get('disease_risk', 0.0)
        
        recommendations = {
            "immediate_action": [],
            "treatment": [],
            "prevention": []
        }
        
        if disease_risk > 0.7:
            recommendations["immediate_action"].extend([
                "Isolate affected plants immediately",
                "Apply fungicide treatment"
            ])
        
        recommendations["prevention"].extend([
            "Regular field scouting",
            "Proper plant spacing",
            "Good air circulation"
        ])
        
        return recommendations
    
    def _generate_pest_recommendations(self, predictions: Dict[str, Any], request: PestPredictionRequest) -> Dict[str, Any]:
        """Generate pest-specific recommendations."""
        pest_risk = predictions.get('pest_risk', 0.0)
        
        recommendations = {
            "immediate_action": [],
            "treatment": [],
            "prevention": []
        }
        
        if pest_risk > 0.6:
            recommendations["immediate_action"].extend([
                "Apply targeted pesticide",
                "Remove heavily infested plants"
            ])
        
        recommendations["prevention"].extend([
            "Regular pest monitoring",
            "Integrated pest management",
            "Biological control methods"
        ])
        
        return recommendations
    
    async def predict_rainfall(self, request: RainfallPredictionRequest) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Predict rainfall using the loaded rainfall model."""
        try:
            # Prepare weather data for rainfall prediction
            weather_data = {
                'year': request.year,
                'subdivision': request.subdivision,
                'month': request.month,
                'current_rainfall': request.current_rainfall
            }
            
            # Get prediction from ML service
            predictions, confidence, recommendations = await self.ml_service.predict_rainfall(weather_data)
            
            logger.info(f"Rainfall prediction completed: {predictions.get('predicted_rainfall', 0):.2f}mm")
            return predictions, confidence, recommendations
            
        except Exception as e:
            logger.error(f"Rainfall prediction error: {str(e)}")
            raise
    
    async def predict_soil_type(self, request: SoilTypePredictionRequest) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """Predict soil type using the loaded soil model."""
        try:
            # Prepare soil data for soil type prediction
            soil_data = {
                'nitrogen': request.nitrogen,
                'phosphorus': request.phosphorus,
                'potassium': request.potassium,
                'temperature': request.temperature,
                'moisture': request.moisture,
                'humidity': request.humidity
            }
            
            # Get prediction from ML service
            predictions, confidence, recommendations = await self.ml_service.predict_soil_type(soil_data)
            
            logger.info(f"Soil type prediction completed: {predictions.get('predicted_soil_type', 'Unknown')}")
            return predictions, confidence, recommendations
            
        except Exception as e:
            logger.error(f"Soil type prediction error: {str(e)}")
            raise

# Global prediction service instance
prediction_service = PredictionService()