from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional
from app.models.schemas import UserCreate, UserLogin, UserResponse, TokenResponse, GoogleAuthRequest, ErrorResponse
from app.services.auth import auth_service
from app.utils.security import create_access_token, get_current_user
from app.database import supabase
import logging

router = APIRouter()  # Define the router
logger = logging.getLogger(__name__)

@router.post("/register", response_model=TokenResponse)
async def register(user_data: UserCreate):
    try:
        # FIXED: Remove await since register_user is not async
        user = auth_service.register_user(user_data.dict(exclude_unset=True))
        access_token = create_access_token({"sub": user["id"]})
        return TokenResponse(access_token=access_token, user=UserResponse(**user), token_type="bearer")
    except HTTPException as e:
        logger.error(f"Registration failed: {str(e.detail)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during registration: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin):
    try:
        # FIXED: Remove await since login_user is not async
        user = auth_service.login_user(user_data.email, user_data.password)
        access_token = create_access_token({"sub": user["id"]})
        return TokenResponse(access_token=access_token, user=UserResponse(**user), token_type="bearer")
    except HTTPException as e:
        logger.error(f"Login failed: {str(e.detail)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during login: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    return UserResponse(**current_user)

# Google OAuth endpoints (commented out for testing without credentials)
# @router.post("/google", response_model=TokenResponse)
# async def google_login(google_request: GoogleAuthRequest):
#     try:
#         idinfo = auth_service.verify_google_token(google_request.token)
#         user = auth_service.create_or_login_google_user(idinfo)
#         access_token = create_access_token({"sub": user["id"]})
#         return TokenResponse(access_token=access_token, user=UserResponse(**user), token_type="bearer")
#     except HTTPException as e:
#         logger.error(f"Google login failed: {str(e.detail)}")
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error during Google login: {str(e)}")
#         raise HTTPException(status_code=500, detail="Internal server error")