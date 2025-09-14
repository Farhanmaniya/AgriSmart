"""
Dashboard API routes for AgriSmart backend.
Handles dashboard statistics and analytics for farmers.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime, timedelta
import logging

from app.models.schemas import DashboardStats, CropAnalytics, PredictionResponse
from app.utils.security import get_current_user
from app.utils.logging import log_request, log_error
from app.database import db_ops

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/stats",
    response_model=DashboardStats,
    summary="Get dashboard statistics",
    description="Get user's dashboard statistics and overview"
)
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    """Get dashboard statistics for user."""
    log_request(logger, "GET", "/api/dashboard/stats", str(current_user["id"]))
    
    try:
        # Get comprehensive dashboard stats
        stats_data = await db_ops.get_dashboard_stats(str(current_user["id"]))
        
        # Convert recent predictions to PredictionResponse objects
        recent_predictions = []
        for pred in stats_data.get("recent_predictions", []):
            try:
                recent_predictions.append(PredictionResponse(**pred))
            except Exception as e:
                logger.warning(f"Error converting prediction to response: {str(e)}")
                continue
        
        # Create dashboard stats response
        dashboard_stats = DashboardStats(
            total_predictions=stats_data.get("total_predictions", 0),
            accuracy_rate=stats_data.get("accuracy_rate", "0%"),
            last_prediction=stats_data.get("last_prediction", "Never"),
            irrigation_count=stats_data.get("irrigation_count", 0),
            member_since=stats_data.get("member_since", 2025),
            recent_predictions=recent_predictions
        )
        
        logger.info(f"Dashboard stats retrieved for user {current_user['id']}")
        return dashboard_stats
        
    except Exception as e:
        log_error(logger, e, "Get dashboard stats")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard statistics"
        )


@router.get(
    "/analytics",
    response_model=list[CropAnalytics],
    summary="Get crop analytics",
    description="Get detailed crop analytics and insights"
)
async def get_crop_analytics(current_user: dict = Depends(get_current_user)):
    """Get crop analytics for user."""
    log_request(logger, "GET", "/api/dashboard/analytics", str(current_user["id"]))
    
    try:
        # Get user's main crops
        main_crops = current_user.get("main_crops", "wheat, rice").lower()
        crop_list = [crop.strip() for crop in main_crops.split(",")]
        
        # Generate analytics for each crop
        analytics = []
        for crop in crop_list[:5]:  # Limit to 5 crops
            crop_analytics = CropAnalytics(
                crop_type=crop.title(),
                total_area=current_user.get("farm_size", 0.0) / len(crop_list),
                avg_yield=_generate_mock_yield(crop),
                disease_incidents=_generate_mock_incidents("disease"),
                pest_incidents=_generate_mock_incidents("pest"),
                irrigation_frequency=_generate_mock_irrigation_frequency()
            )
            analytics.append(crop_analytics)
        
        logger.info(f"Crop analytics generated for user {current_user['id']}")
        return analytics
        
    except Exception as e:
        log_error(logger, e, "Get crop analytics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve crop analytics"
        )


@router.get(
    "/overview",
    response_model=dict,
    summary="Get dashboard overview",
    description="Get comprehensive dashboard overview with key metrics"
)
async def get_dashboard_overview(current_user: dict = Depends(get_current_user)):
    """Get dashboard overview."""
    log_request(logger, "GET", "/api/dashboard/overview", str(current_user["id"]))
    
    try:
        # Get basic stats
        stats_data = await db_ops.get_dashboard_stats(str(current_user["id"]))
        
        # Calculate additional metrics
        overview = {
            "user_profile": {
                "name": current_user.get("name"),
                "region": current_user.get("region"),
                "farm_size": current_user.get("farm_size"),
                "main_crops": current_user.get("main_crops"),
                "member_since": current_user.get("member_since", 2025)
            },
            "quick_stats": {
                "total_predictions": stats_data.get("total_predictions", 0),
                "accuracy_rate": stats_data.get("accuracy_rate", "0%"),
                "irrigation_schedules": stats_data.get("irrigation_count", 0),
                "active_alerts": _generate_mock_alerts_count()
            },
            "recent_activity": {
                "last_prediction": stats_data.get("last_prediction", "Never"),
                "last_irrigation": _get_last_irrigation_date(),
                "predictions_this_month": _calculate_monthly_predictions(stats_data.get("recent_predictions", [])),
                "water_saved": _calculate_water_saved()
            },
            "recommendations": [
                "Consider switching to drip irrigation for better efficiency",
                "Monitor weather forecast for upcoming rainfall",
                "Schedule disease prediction for tomato crop",
                "Review irrigation schedule for next week"
            ],
            "alerts": _generate_mock_alerts()
        }
        
        logger.info(f"Dashboard overview generated for user {current_user['id']}")
        return overview
        
    except Exception as e:
        log_error(logger, e, "Get dashboard overview")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard overview"
        )


@router.get(
    "/performance",
    response_model=dict,
    summary="Get performance metrics",
    description="Get detailed performance metrics and trends"
)
async def get_performance_metrics(current_user: dict = Depends(get_current_user)):
    """Get performance metrics."""
    log_request(logger, "GET", "/api/dashboard/performance", str(current_user["id"]))
    
    try:
        # Generate performance metrics
        performance = {
            "prediction_accuracy": {
                "overall": 85.5,
                "yield_predictions": 88.2,
                "disease_detection": 82.1,
                "pest_classification": 86.7,
                "trend": "improving"
            },
            "irrigation_efficiency": {
                "water_saved_percentage": 23.5,
                "cost_savings": 1250.75,
                "efficiency_score": 78.9,
                "trend": "stable"
            },
            "crop_health": {
                "overall_score": 82.3,
                "disease_prevention": 89.1,
                "pest_control": 75.6,
                "yield_optimization": 84.2
            },
            "monthly_trends": _generate_monthly_trends(),
            "comparisons": {
                "regional_average": 72.1,
                "your_performance": 85.5,
                "improvement": 13.4
            }
        }
        
        logger.info(f"Performance metrics generated for user {current_user['id']}")
        return performance
        
    except Exception as e:
        log_error(logger, e, "Get performance metrics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )


@router.get(
    "/insights",
    response_model=dict,
    summary="Get AI insights",
    description="Get AI-generated insights and recommendations"
)
async def get_ai_insights(current_user: dict = Depends(get_current_user)):
    """Get AI-powered insights."""
    log_request(logger, "GET", "/api/dashboard/insights", str(current_user["id"]))
    
    try:
        # Generate AI insights
        insights = {
            "key_insights": [
                {
                    "title": "Optimal Irrigation Timing",
                    "description": "Your crops show 15% better yield when irrigated between 6-8 AM",
                    "impact": "high",
                    "action": "Schedule morning irrigation"
                },
                {
                    "title": "Disease Risk Alert",
                    "description": "High humidity levels increase fungal disease risk by 30%",
                    "impact": "medium",
                    "action": "Apply preventive fungicide"
                },
                {
                    "title": "Nutrient Optimization",
                    "description": "Nitrogen levels in field 2 are 20% below optimal",
                    "impact": "medium",
                    "action": "Apply nitrogen-rich fertilizer"
                }
            ],
            "predictive_alerts": [
                {
                    "type": "weather",
                    "message": "Heavy rainfall expected in 3 days - adjust irrigation schedule",
                    "severity": "medium",
                    "expires_at": (datetime.now() + timedelta(days=3)).isoformat()
                },
                {
                    "type": "pest",
                    "message": "Aphid activity likely to increase with rising temperatures",
                    "severity": "low",
                    "expires_at": (datetime.now() + timedelta(days=7)).isoformat()
                }
            ],
            "optimization_suggestions": [
                "Switch to drought-resistant crop varieties for 20% water savings",
                "Implement integrated pest management for cost-effective pest control",
                "Use cover crops to improve soil health and reduce irrigation needs"
            ],
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info(f"AI insights generated for user {current_user['id']}")
        return insights
        
    except Exception as e:
        log_error(logger, e, "Get AI insights")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate AI insights"
        )


# Helper functions for mock data generation
def _generate_mock_yield(crop: str) -> float:
    """Generate mock yield data for crop."""
    base_yields = {
        "wheat": 3.2,
        "rice": 4.8,
        "corn": 5.5,
        "cotton": 2.1,
        "sugarcane": 75.2,
        "tomato": 45.3,
        "potato": 28.5,
        "onion": 32.1
    }
    return base_yields.get(crop.lower(), 3.0)


def _generate_mock_incidents(incident_type: str) -> int:
    """Generate mock incident count."""
    import random
    if incident_type == "disease":
        return random.randint(0, 3)
    elif incident_type == "pest":
        return random.randint(0, 5)
    return 0


def _generate_mock_irrigation_frequency() -> float:
    """Generate mock irrigation frequency."""
    import random
    return round(random.uniform(2.0, 7.0), 1)


def _generate_mock_alerts_count() -> int:
    """Generate mock alerts count."""
    import random
    return random.randint(0, 4)


def _get_last_irrigation_date() -> str:
    """Get mock last irrigation date."""
    last_date = datetime.now() - timedelta(days=2)
    return last_date.strftime("%Y-%m-%d")


def _calculate_monthly_predictions(predictions: list) -> int:
    """Calculate predictions made this month."""
    current_month = datetime.now().month
    current_year = datetime.now().year
    
    monthly_count = 0
    for pred in predictions:
        try:
            pred_date = datetime.fromisoformat(pred.get("created_at", "").replace('Z', '+00:00'))
            if pred_date.month == current_month and pred_date.year == current_year:
                monthly_count += 1
        except:
            continue
    
    return monthly_count


def _calculate_water_saved() -> float:
    """Calculate mock water saved."""
    import random
    return round(random.uniform(500, 3000), 2)


def _generate_mock_alerts() -> list:
    """Generate mock alerts."""
    return [
        {
            "id": "alert_1",
            "type": "weather",
            "message": "Heavy rainfall expected tomorrow",
            "severity": "medium",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "alert_2", 
            "type": "irrigation",
            "message": "Soil moisture below optimal level",
            "severity": "low",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat()
        }
    ]


def _generate_monthly_trends() -> dict:
    """Generate mock monthly trends."""
    import random
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    predictions = [random.randint(5, 25) for _ in months]
    accuracy = [random.uniform(75, 95) for _ in months]
    water_saved = [random.uniform(1000, 4000) for _ in months]
    
    return {
        "months": months,
        "predictions": predictions,
        "accuracy": [round(a, 1) for a in accuracy],
        "water_saved": [round(w, 1) for w in water_saved]
    }