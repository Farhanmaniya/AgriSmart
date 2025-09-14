# ===== app/database.py (CORRECTED - Added missing method) =====
"""
Database configuration and utilities for AgriSmart Backend.
Uses Supabase (PostgreSQL) for data storage with Row Level Security.
"""

import os
from typing import Optional, Dict, Any, List
from supabase import create_client, Client
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Supabase client instance
supabase: Optional[Client] = None

def get_supabase_client() -> Client:
    """Get Supabase client instance."""
    global supabase
    
    if supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")
        
        supabase = create_client(url, key)
        logger.info("Supabase client initialized")
    
    return supabase

async def init_database():
    """Initialize database connection and ensure tables exist."""
    client = get_supabase_client()
    
    try:
        # Test connection by fetching from auth.users (system table)
        result = client.auth.get_session()
        logger.info("Database connection successful")
        
        # Initialize ML models directory if needed
        await _ensure_ml_models_directory()
        
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

async def _ensure_ml_models_directory():
    """Ensure ML models directory exists."""
    import os
    models_dir = "app/ml_models/saved_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        logger.info("ML models directory created")

class DatabaseOperations:
    """Database operations wrapper for Supabase."""
    
    def __init__(self):
        self.client = get_supabase_client()
    
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user in the database."""
        try:
            result = self.client.table("users").insert(user_data).execute()
            if result.data:
                logger.info(f"User created successfully: {user_data.get('email')}")
                return result.data[0]
            else:
                raise Exception("Failed to create user")
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            raise
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email address."""
        try:
            result = self.client.table("users").select("*").eq("email", email).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching user by email: {str(e)}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        try:
            result = self.client.table("users").select("*").eq("id", user_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching user by ID: {str(e)}")
            return None
    
    async def update_user_stats(self, user_id: str, predictions_count: int, 
                              accuracy_rate: str, last_prediction: str) -> bool:
        """Update user prediction statistics."""
        try:
            result = self.client.table("users").update({
                "predictions_count": predictions_count,
                "accuracy_rate": accuracy_rate,
                "last_prediction": last_prediction
            }).eq("id", user_id).execute()
            return bool(result.data)
        except Exception as e:
            logger.error(f"Error updating user stats: {str(e)}")
            return False
    
    async def create_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store prediction result in database."""
        try:
            result = self.client.table("predictions").insert(prediction_data).execute()
            if result.data:
                logger.info(f"Prediction stored for user: {prediction_data.get('user_id')}")
                return result.data[0]
            else:
                raise Exception("Failed to store prediction")
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
            raise
    
    async def get_user_predictions(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's recent predictions."""
        try:
            result = self.client.table("predictions").select("*").eq(
                "user_id", user_id
            ).order("created_at", desc=True).limit(limit).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error fetching user predictions: {str(e)}")
            return []
    
    # ADDED MISSING METHOD
    async def get_prediction_by_id(self, prediction_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get specific prediction by ID for a user."""
        try:
            result = self.client.table("predictions").select("*").eq(
                "id", prediction_id
            ).eq("user_id", user_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching prediction by ID: {str(e)}")
            return None
    
    async def create_irrigation_log(self, irrigation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store irrigation schedule in database."""
        try:
            result = self.client.table("irrigation_logs").insert(irrigation_data).execute()
            if result.data:
                logger.info(f"Irrigation log stored for user: {irrigation_data.get('user_id')}")
                return result.data[0]
            else:
                raise Exception("Failed to store irrigation log")
        except Exception as e:
            logger.error(f"Error storing irrigation log: {str(e)}")
            raise
    
    async def get_user_irrigation_logs(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's recent irrigation logs."""
        try:
            result = self.client.table("irrigation_logs").select("*").eq(
                "user_id", user_id
            ).order("created_at", desc=True).limit(limit).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error fetching irrigation logs: {str(e)}")
            return []
    
    async def get_dashboard_stats(self, user_id: str) -> Dict[str, Any]:
        """Get dashboard statistics for user."""
        try:
            # Get user info
            user = await self.get_user_by_id(user_id)
            if not user:
                return {}
            
            # Get recent predictions
            predictions = await self.get_user_predictions(user_id, 5)
            
            # Get irrigation count
            irrigation_result = self.client.table("irrigation_logs").select(
                "id", count="exact"
            ).eq("user_id", user_id).execute()
            
            irrigation_count = irrigation_result.count or 0
            
            return {
                "total_predictions": user.get("predictions_count", 0),
                "accuracy_rate": user.get("accuracy_rate", "0%"),
                "last_prediction": user.get("last_prediction", "Never"),
                "irrigation_count": irrigation_count,
                "recent_predictions": predictions,
                "member_since": user.get("member_since", 2025)
            }
        except Exception as e:
            logger.error(f"Error fetching dashboard stats: {str(e)}")
            return {}

# Global database operations instance
db_ops = DatabaseOperations()
