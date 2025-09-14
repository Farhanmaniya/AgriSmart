from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta, timezone
from app.database import supabase
from app.utils.security import SECRET_KEY, ALGORITHM
from fastapi import HTTPException, status
import os
from uuid import uuid4
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    """Authentication service for user management."""
    
    def hash_password(self, password: str) -> str:
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, data: dict, expires_delta: timedelta = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    # Google OAuth methods (commented out for testing without credentials)
    # def verify_google_token(self, token: str) -> dict:
    #     CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    #     try:
    #         from google.oauth2 import id_token
    #         from google.auth.transport import requests
    #         idinfo = id_token.verify_oauth2_token(token, requests.Request(), CLIENT_ID)
    #         if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
    #             raise ValueError("Wrong issuer.")
    #         return idinfo
    #     except ValueError:
    #         raise HTTPException(status_code=400, detail="Invalid Google token")

    def register_user(self, user_data: dict) -> dict:
        # Ensure all required fields are present with defaults if missing
        user_data = {
            'name': user_data.get('name', ''),
            'email': user_data.get('email', ''),
            'phone': user_data.get('phone', ''),
            'region': user_data.get('region', ''),
            'farm_size': user_data.get('farm_size', 0.0),
            'main_crops': user_data.get('main_crops', ''),
            'password': self.hash_password(user_data.pop('password', ''))  # Required, hash it
        }
        if not all([user_data['name'], user_data['email'], user_data['phone'], user_data['region'], 
                    user_data['main_crops'], user_data['password']]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        if user_data['farm_size'] < 0:
            raise HTTPException(status_code=400, detail="Farm size must be non-negative")

        user_data.update({
            'id': str(uuid4()),
            'member_since': int(datetime.now(timezone.utc).timestamp()),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'predictions_count': 0,
            'accuracy_rate': '0%',
            'last_prediction': 'Never'
        })

        logger.info(f"Inserting user data: {user_data}")
        response = supabase.table('users').insert(user_data).execute()
        if response.data:
            logger.info(f"User registered: {response.data[0]}")
            return response.data[0]
        error_detail = response.error.message if response.error else "Unknown error"
        logger.error(f"Supabase insert failed: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Failed to save user to Supabase: {error_detail}")

    def login_user(self, email: str, password: str) -> dict:
        response = supabase.table('users').select('*').eq('email', email).execute()
        if not response.data or not self.verify_password(password, response.data[0]['password']):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return response.data[0]

    def get_user_by_email(self, email: str) -> Optional[dict]:
        response = supabase.table('users').select('*').eq('email', email).execute()
        return response.data[0] if response.data else None

    # Google OAuth method (commented out for testing)
    # def create_or_login_google_user(self, idinfo: dict) -> dict:
    #     user = self.get_user_by_email(idinfo['email'])
    #     if not user:
    #         user_data = {
    #             'id': str(uuid4()),
    #             'name': idinfo['name'],
    #             'email': idinfo['email'],
    #             'phone': '',  # Placeholder
    #             'region': '',
    #             'farm_size': 0.0,
    #             'main_crops': '',
    #             'password': '',  # No password for OAuth
    #             'member_since': int(datetime.now(timezone.utc).timestamp()),
    #             'created_at': datetime.now(timezone.utc).isoformat(),
    #             'predictions_count': 0,
    #             'accuracy_rate': '0%',
    #             'last_prediction': 'Never'
    #         }
    #         response = supabase.table('users').insert(user_data).execute()
    #         user = response.data[0]
    #     return user

# Export the service instance
auth_service = AuthService()