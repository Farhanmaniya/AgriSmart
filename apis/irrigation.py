"""
Irrigation API routes for AgriSmart backend.
Handles irrigation scheduling and water management.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime
import uuid
import logging

from app.models.schemas import IrrigationRequest, IrrigationResponse
from app.services.irrigation import irrigation_service
from app.utils.security import get_current_user
from app.utils.logging import log_request, log_error, log_irrigation_schedule
from app.database import db_ops

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/schedule",
    response_model=IrrigationResponse,
    summary="Calculate irrigation schedule",
    description="Calculate optimal irrigation schedule based on conditions"
)
async def calculate_irrigation_schedule(
    request: IrrigationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Calculate irrigation schedule for user's crop."""
    log_request(logger, "POST", "/api/irrigation/schedule", str(current_user["id"]))
    
    try:
        # Get irrigation schedule from service
        schedule = await irrigation_service.calculate_irrigation_schedule(request)
        
        # Only log irrigation if it's actually scheduled (duration > 0)
        if schedule.duration_minutes > 0:
            # Prepare data for database
            irrigation_data = {
                "id": str(uuid.uuid4()),
                "user_id": str(current_user["id"]),
                "schedule_date": schedule.schedule_date,
                "duration_minutes": schedule.duration_minutes,
                "water_volume": schedule.water_volume,
                "weather_data": {
                    "temperature": request.temperature,
                    "rainfall": request.rainfall,
                    "soil_moisture": request.soil_moisture,
                    "crop_type": request.crop_type.value,
                    "area": request.area
                }
            }
            
            # Store irrigation log in database
            await db_ops.create_irrigation_log(irrigation_data)
            
            # Log successful irrigation scheduling
            log_irrigation_schedule(
                logger, 
                str(current_user["id"]), 
                schedule.duration_minutes, 
                schedule.water_volume
            )
        
        logger.info(f"Irrigation schedule calculated for user {current_user['id']}")
        
        return schedule
        
    except Exception as e:
        log_error(logger, e, "Irrigation scheduling")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Irrigation scheduling failed"
        )


@router.get(
    "/history",
    response_model=list[dict],
    summary="Get irrigation history",
    description="Get user's irrigation history and logs"
)
async def get_irrigation_history(
    days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    """Get user's irrigation history."""
    log_request(logger, "GET", "/api/irrigation/history", str(current_user["id"]))
    
    try:
        # Get irrigation logs from database
        logs = await db_ops.get_user_irrigation_logs(str(current_user["id"]), days)
        
        # Convert to response format
        history = []
        for log in logs:
            history.append({
                "id": log["id"],
                "schedule_date": log["schedule_date"],
                "duration_minutes": log["duration_minutes"],
                "water_volume": log["water_volume"],
                "weather_data": log["weather_data"],
                "created_at": log["created_at"]
            })
        
        logger.info(f"Retrieved {len(history)} irrigation logs for user {current_user['id']}")
        return history
        
    except Exception as e:
        log_error(logger, e, "Get irrigation history")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve irrigation history"
        )


@router.get(
    "/recommendations",
    response_model=dict,
    summary="Get irrigation recommendations",
    description="Get general irrigation recommendations for user's region"
)
async def get_irrigation_recommendations(
    crop_type: str = None,
    current_user: dict = Depends(get_current_user)
):
    """Get irrigation recommendations."""
    log_request(logger, "GET", "/api/irrigation/recommendations", str(current_user["id"]))
    
    try:
        # Generate general recommendations
        recommendations = {
            "general": [
                "Monitor soil moisture levels regularly",
                "Irrigate early morning or late evening",
                "Use drip irrigation for water efficiency",
                "Install mulch to retain soil moisture"
            ],
            "seasonal": {
                "summer": [
                    "Increase irrigation frequency during hot weather",
                    "Provide shade during peak heat hours",
                    "Check for signs of heat stress"
                ],
                "winter": [
                    "Reduce irrigation frequency in cool weather",
                    "Avoid overwatering in low temperatures",
                    "Monitor for frost conditions"
                ],
                "monsoon": [
                    "Reduce irrigation during heavy rainfall",
                    "Ensure proper drainage",
                    "Monitor for waterlogging"
                ]
            },
            "efficiency_tips": [
                "Use soil moisture sensors",
                "Implement automated irrigation systems",
                "Regular maintenance of irrigation equipment",
                "Track water usage and costs"
            ]
        }
        
        # Add crop-specific recommendations if crop type provided
        if crop_type:
            crop_recommendations = {
                "wheat": ["Deep, less frequent irrigation", "Critical stages: tillering, flowering"],
                "rice": ["Maintain standing water", "Drain before harvest"],
                "corn": ["Critical periods: silking, grain filling", "Avoid water stress during tasseling"],
                "cotton": ["Deep irrigation", "Reduce water during boll opening"]
            }
            
            if crop_type.lower() in crop_recommendations:
                recommendations["crop_specific"] = crop_recommendations[crop_type.lower()]
        
        return {
            "recommendations": recommendations,
            "user_region": current_user.get("region", "India"),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        log_error(logger, e, "Get irrigation recommendations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recommendations"
        )


@router.get(
    "/efficiency",
    response_model=dict,
    summary="Get irrigation efficiency analysis",
    description="Analyze irrigation efficiency and water usage patterns"
)
async def get_irrigation_efficiency(
    current_user: dict = Depends(get_current_user)
):
    """Get irrigation efficiency analysis."""
    log_request(logger, "GET", "/api/irrigation/efficiency", str(current_user["id"]))
    
    try:
        # Get efficiency analysis from service
        analysis = await irrigation_service.analyze_irrigation_efficiency(str(current_user["id"]))
        
        logger.info(f"Irrigation efficiency analysis generated for user {current_user['id']}")
        return analysis
        
    except Exception as e:
        log_error(logger, e, "Get irrigation efficiency")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze irrigation efficiency"
        )


@router.post(
    "/feedback",
    response_model=dict,
    summary="Submit irrigation feedback",
    description="Submit feedback on irrigation effectiveness"
)
async def submit_irrigation_feedback(
    feedback_data: dict,
    current_user: dict = Depends(get_current_user)
):
    """Submit irrigation feedback."""
    log_request(logger, "POST", "/api/irrigation/feedback", str(current_user["id"]))
    
    try:
        irrigation_id = feedback_data.get("irrigation_id")
        effectiveness = feedback_data.get("effectiveness")  # "high", "medium", "low"
        comments = feedback_data.get("comments", "")
        
        if not irrigation_id or not effectiveness:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="irrigation_id and effectiveness are required"
            )
        
        # Store feedback (would be implemented with database)
        feedback_record = {
            "user_id": str(current_user["id"]),
            "irrigation_id": irrigation_id,
            "effectiveness": effectiveness,
            "comments": comments,
            "submitted_at": datetime.now().isoformat()
        }
        
        logger.info(f"Irrigation feedback submitted by user {current_user['id']}")
        
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": str(uuid.uuid4()),
            "status": "received"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(logger, e, "Submit irrigation feedback")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )


@router.get(
    "/weather-impact",
    response_model=dict,
    summary="Get weather impact analysis",
    description="Analyze how weather conditions affect irrigation needs"
)
async def get_weather_impact_analysis(
    current_user: dict = Depends(get_current_user)
):
    """Get weather impact analysis for irrigation."""
    log_request(logger, "GET", "/api/irrigation/weather-impact", str(current_user["id"]))
    
    try:
        # Mock weather impact analysis
        analysis = {
            "current_conditions": {
                "temperature": 28.5,
                "humidity": 65,
                "rainfall_24h": 5.2,
                "wind_speed": 8.5,
                "impact_score": 7.2  # Out of 10
            },
            "irrigation_adjustments": {
                "frequency_modifier": 1.1,  # 10% more frequent
                "duration_modifier": 0.9,   # 10% shorter duration
                "reason": "Moderate temperature with low recent rainfall"
            },
            "weekly_forecast_impact": [
                {"day": "Today", "irrigation_need": "Medium", "modifier": 1.0},
                {"day": "Tomorrow", "irrigation_need": "High", "modifier": 1.2},
                {"day": "Day 3", "irrigation_need": "Low", "modifier": 0.7},
                {"day": "Day 4", "irrigation_need": "Low", "modifier": 0.6},
                {"day": "Day 5", "irrigation_need": "Medium", "modifier": 1.0}
            ],
            "recommendations": [
                "Monitor soil moisture closely due to increasing temperatures",
                "Reduce irrigation duration but maintain frequency",
                "Consider mulching to retain soil moisture"
            ]
        }
        
        logger.info(f"Weather impact analysis generated for user {current_user['id']}")
        return analysis
        
    except Exception as e:
        log_error(logger, e, "Get weather impact analysis")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate weather impact analysis"
        )