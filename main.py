"""
AgriSmart Backend - Smart Irrigation and Crop Monitoring System
SIH25044 - Smart India Hackathon 2025

FastAPI application for AI-powered agriculture management system.
Handles authentication, ML predictions, irrigation scheduling, and analytics.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn
import os
from dotenv import load_dotenv

from app.apis import auth, predictions, irrigation, dashboard
from app.utils.logging import setup_logger
from app.database import init_database

# Load environment variables
load_dotenv()

# Initialize logger
logger = setup_logger(__name__)

# Create FastAPI instance
app = FastAPI(
    title="AgriSmart API",
    description="AI-powered agriculture management system for Smart India Hackathon 2025",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS Configuration
origins = [
    "https://agrismart-phi.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(irrigation.router, prefix="/api/irrigation", tags=["Irrigation"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting AgriSmart Backend...")
    
    # Initialize database and ML models
    try:
        await init_database()
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down AgriSmart Backend...")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AgriSmart API - Smart Irrigation and Crop Monitoring System",
        "version": "1.0.0",
        "project": "SIH25044",
        "hackathon": "Smart India Hackathon 2025",
        "documentation": "/api/docs",
        "endpoints": {
            "auth": "/api/auth",
            "predictions": "/api/predictions", 
            "irrigation": "/api/irrigation",
            "dashboard": "/api/dashboard"
        },
        "ml_models": {
            "pest_detection": "pest_model.h5 (TensorFlow/Keras)",
            "rainfall_prediction": "rainfall_model.joblib (scikit-learn)",
            "soil_classification": "soil_model.joblib (scikit-learn)"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "database": "connected"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )