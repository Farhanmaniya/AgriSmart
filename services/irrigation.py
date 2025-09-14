"""
Irrigation service for AgriSmart backend.
Handles irrigation scheduling based on weather data, soil moisture, and crop requirements.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from app.models.schemas import IrrigationRequest, IrrigationResponse, CropType

logger = logging.getLogger(__name__)


class IrrigationService:
    """Irrigation scheduling and management service."""
    
    def __init__(self):
        # Crop water requirements (mm/day during growing season)
        self.crop_water_requirements = {
            CropType.WHEAT: 4.5,
            CropType.RICE: 8.0,
            CropType.CORN: 6.0,
            CropType.COTTON: 5.5,
            CropType.SUGARCANE: 7.5,
            CropType.TOMATO: 5.0,
            CropType.POTATO: 4.0,
            CropType.ONION: 3.5
        }
        
        # Soil moisture thresholds
        self.moisture_thresholds = {
            "critical": 20,    # Below this: immediate irrigation needed
            "low": 30,         # Below this: irrigation recommended
            "optimal": 60,     # Target moisture level
            "high": 80         # Above this: no irrigation needed
        }
        
        # Weather impact factors
        self.temperature_factors = {
            "hot": (35, 1.3),      # Above 35°C: 30% more water
            "warm": (25, 1.0),     # 25-35°C: normal water
            "cool": (15, 0.8),     # 15-25°C: 20% less water
            "cold": (0, 0.6)       # Below 15°C: 40% less water
        }
    
    async def calculate_irrigation_schedule(self, request: IrrigationRequest) -> IrrigationResponse:
        """Calculate irrigation schedule based on input parameters."""
        try:
            # Get crop water requirement
            base_water_req = self.crop_water_requirements.get(request.crop_type, 5.0)
            
            # Assess current conditions
            irrigation_needed = self._assess_irrigation_need(request)
            
            if not irrigation_needed:
                return self._create_no_irrigation_response(request)
            
            # Calculate irrigation parameters
            duration = self._calculate_duration(request, base_water_req)
            water_volume = self._calculate_water_volume(request, duration)
            schedule_time = self._determine_optimal_time(request)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(request)
            
            # Calculate next irrigation
            next_irrigation = self._calculate_next_irrigation(request, duration)
            
            response = IrrigationResponse(
                schedule_date=schedule_time,
                duration_minutes=duration,
                water_volume=water_volume,
                recommendations=recommendations,
                next_irrigation=next_irrigation
            )
            
            logger.info(f"Irrigation scheduled - Duration: {duration}min, Volume: {water_volume}L")
            
            return response
            
        except Exception as e:
            logger.error(f"Irrigation calculation error: {str(e)}")
            raise Exception("Irrigation scheduling failed")
    
    def _assess_irrigation_need(self, request: IrrigationRequest) -> bool:
        """Assess if irrigation is needed based on conditions."""
        # Check soil moisture
        if request.soil_moisture >= self.moisture_thresholds["high"]:
            return False  # Soil is too wet
        
        # Check recent rainfall
        if request.rainfall > 15:  # More than 15mm recent rainfall
            return False
        
        # Check if irrigation is critically needed
        if request.soil_moisture < self.moisture_thresholds["critical"]:
            return True  # Critical irrigation needed
        
        # Check if irrigation is recommended
        if request.soil_moisture < self.moisture_thresholds["low"]:
            # Consider temperature
            if request.temperature > 30:
                return True  # Hot weather, irrigation needed
            elif request.rainfall < 5:  # Less than 5mm rainfall
                return True
        
        return False
    
    def _create_no_irrigation_response(self, request: IrrigationRequest) -> IrrigationResponse:
        """Create response when no irrigation is needed."""
        next_check = datetime.now() + timedelta(days=2)
        
        recommendations = {
            "message": "No irrigation needed at this time",
            "reasons": [],
            "next_check": next_check.isoformat()
        }
        
        if request.soil_moisture >= self.moisture_thresholds["high"]:
            recommendations["reasons"].append("Soil moisture is adequate")
        
        if request.rainfall > 15:
            recommendations["reasons"].append("Recent rainfall is sufficient")
        
        return IrrigationResponse(
            schedule_date=next_check,
            duration_minutes=0,
            water_volume=0.0,
            recommendations=recommendations,
            next_irrigation=next_check
        )
    
    def _calculate_duration(self, request: IrrigationRequest, base_water_req: float) -> int:
        """Calculate irrigation duration in minutes."""
        # Base duration calculation
        moisture_deficit = max(0, self.moisture_thresholds["optimal"] - request.soil_moisture)
        base_duration = (moisture_deficit / 100) * 60  # Base: 1 hour for 100% deficit
        
        # Adjust for crop water requirements
        crop_factor = base_water_req / 5.0  # Normalize to average requirement
        duration = base_duration * crop_factor
        
        # Adjust for temperature
        temp_factor = self._get_temperature_factor(request.temperature)
        duration *= temp_factor
        
        # Adjust for area
        if request.area > 1.0:
            duration *= min(2.0, 1 + (request.area - 1) * 0.2)  # Max 2x for large areas
        
        # Ensure reasonable bounds
        duration = max(15, min(120, int(duration)))  # 15 min to 2 hours
        
        return duration
    
    def _calculate_water_volume(self, request: IrrigationRequest, duration: int) -> float:
        """Calculate water volume in liters."""
        # Base water rate: 10 L/min per hectare
        base_rate = 10.0  # L/min/hectare
        
        # Adjust rate based on crop type
        crop_water_req = self.crop_water_requirements.get(request.crop_type, 5.0)
        rate_factor = crop_water_req / 5.0
        
        # Calculate volume
        volume = request.area * base_rate * (duration / 60) * rate_factor
        
        # Adjust for soil moisture deficit
        moisture_factor = max(0.5, (self.moisture_thresholds["optimal"] - request.soil_moisture) / 50)
        volume *= moisture_factor
        
        return round(volume, 2)
    
    def _determine_optimal_time(self, request: IrrigationRequest) -> datetime:
        """Determine optimal time for irrigation."""
        now = datetime.now()
        
        # If urgent (critical moisture), schedule immediately
        if request.soil_moisture < self.moisture_thresholds["critical"]:
            return now
        
        # If hot weather, schedule for early morning
        if request.temperature > 30:
            # Schedule for 6 AM tomorrow
            tomorrow = now + timedelta(days=1)
            optimal_time = tomorrow.replace(hour=6, minute=0, second=0, microsecond=0)
        else:
            # Schedule for evening (6 PM today or tomorrow)
            if now.hour < 18:
                optimal_time = now.replace(hour=18, minute=0, second=0, microsecond=0)
            else:
                tomorrow = now + timedelta(days=1)
                optimal_time = tomorrow.replace(hour=18, minute=0, second=0, microsecond=0)
        
        return optimal_time
    
    def _generate_recommendations(self, request: IrrigationRequest) -> Dict[str, Any]:
        """Generate irrigation recommendations."""
        recommendations = {
            "timing": [],
            "efficiency": [],
            "monitoring": [],
            "general": []
        }
        
        # Timing recommendations
        if request.temperature > 30:
            recommendations["timing"].append("Irrigate early morning (5-7 AM) to reduce evaporation")
        elif request.temperature < 15:
            recommendations["timing"].append("Irrigate during warmer midday hours")
        else:
            recommendations["timing"].append("Evening irrigation (6-8 PM) is optimal")
        
        # Efficiency recommendations
        if request.soil_moisture < self.moisture_thresholds["critical"]:
            recommendations["efficiency"].append("Use drip irrigation for water conservation")
            recommendations["efficiency"].append("Apply mulch to retain moisture")
        
        if request.area > 2.0:
            recommendations["efficiency"].append("Consider automated irrigation system")
        
        # Monitoring recommendations
        recommendations["monitoring"].append("Check soil moisture after 24 hours")
        recommendations["monitoring"].append("Monitor weather forecast for rainfall")
        
        if request.rainfall < 10:
            recommendations["monitoring"].append("Install soil moisture sensors for better monitoring")
        
        # General recommendations
        if request.temperature > 35:
            recommendations["general"].append("Provide shade during peak heat hours")
        
        recommendations["general"].append("Maintain proper drainage to prevent waterlogging")
        
        return recommendations
    
    def _calculate_next_irrigation(self, request: IrrigationRequest, duration: int) -> datetime:
        """Calculate when next irrigation will be needed."""
        # Base interval based on crop type and season
        crop_interval = {
            CropType.WHEAT: 5,      # 5 days
            CropType.RICE: 2,       # 2 days (water-intensive)
            CropType.CORN: 4,       # 4 days
            CropType.COTTON: 6,     # 6 days
            CropType.SUGARCANE: 3,  # 3 days
            CropType.TOMATO: 3,     # 3 days
            CropType.POTATO: 4,     # 4 days
            CropType.ONION: 5       # 5 days
        }.get(request.crop_type, 4)
        
        # Adjust interval based on conditions
        if request.temperature > 30:
            crop_interval = max(1, crop_interval - 1)  # More frequent in hot weather
        elif request.temperature < 20:
            crop_interval += 1  # Less frequent in cool weather
        
        # Consider rainfall forecast (simplified)
        if request.rainfall > 10:
            crop_interval += 2  # Extend interval if recent rainfall
        
        # Calculate next irrigation date
        next_irrigation = datetime.now() + timedelta(days=crop_interval)
        
        return next_irrigation
    
    def _get_temperature_factor(self, temperature: float) -> float:
        """Get temperature adjustment factor for irrigation."""
        if temperature > 35:
            return 1.3  # 30% more water for hot weather
        elif temperature > 25:
            return 1.0  # Normal water requirement
        elif temperature > 15:
            return 0.8  # 20% less water for cool weather
        else:
            return 0.6  # 40% less water for cold weather
    
    async def get_irrigation_history(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get irrigation history for analysis."""
        # This would typically fetch from database
        # For now, return mock data structure
        return [
            {
                "date": (datetime.now() - timedelta(days=i)).isoformat(),
                "duration": 45 + (i % 20),
                "water_volume": 150.5 + (i * 10),
                "crop_type": "wheat",
                "effectiveness": "high" if i % 3 == 0 else "medium"
            }
            for i in range(min(days, 10))
        ]
    
    async def analyze_irrigation_efficiency(self, user_id: str) -> Dict[str, Any]:
        """Analyze irrigation efficiency and provide insights."""
        # Mock efficiency analysis
        return {
            "water_usage_efficiency": 78.5,  # percentage
            "cost_savings": 1250.0,  # in currency
            "recommendations": [
                "Switch to drip irrigation for 15% water savings",
                "Install soil moisture sensors",
                "Optimize irrigation timing based on weather"
            ],
            "monthly_trends": {
                "water_saved": 2500,  # liters
                "cost_saved": 450,    # currency
                "efficiency_improvement": 12  # percentage
            }
        }


# Global irrigation service instance
irrigation_service = IrrigationService()