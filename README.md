# AgriSmart Backend - Smart Irrigation and Crop Monitoring System

[![SIH25044](https://img.shields.io/badge/SIH-2025-blue.svg)](https://www.sih.gov.in/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.2-green.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org/)
[![Supabase](https://img.shields.io/badge/Supabase-2.9.0-green.svg)](https://supabase.com/)

## Project Overview

AgriSmart is an AI-powered agriculture management system developed for **Smart India Hackathon 2025 (SIH25044)**. This backend provides comprehensive APIs for crop monitoring, yield prediction, disease detection, pest classification, and intelligent irrigation scheduling. Designed for small to medium-scale farmers in India, it addresses critical challenges like water conservation and crop health monitoring.

## 🌟 Key Features

- **🔐 Authentication**: JWT + Google OAuth integration
- **🤖 ML Predictions**: Crop yield, disease detection, pest classification  
- **💧 Smart Irrigation**: Weather-based irrigation scheduling
- **📊 Analytics Dashboard**: Comprehensive farming insights
- **🔄 Real-time Monitoring**: Crop health and environmental tracking
- **📱 Multi-platform Support**: REST API for web/mobile clients

## 🏗️ Architecture

```
app/
├── main.py                 # FastAPI application entry point
├── database.py            # Supabase client and operations
├── models/
│   └── schemas.py         # Pydantic models for API validation
├── services/
│   ├── auth.py           # Authentication service
│   ├── ml.py             # Machine learning service  
│   └── irrigation.py     # Irrigation scheduling service
├── apis/                 # API route handlers
│   ├── auth.py           # Authentication endpoints
│   ├── predictions.py    # ML prediction endpoints
│   ├── irrigation.py     # Irrigation endpoints
│   └── dashboard.py      # Dashboard and analytics endpoints
├── utils/
│   ├── security.py       # JWT and security utilities
│   └── logging.py        # Logging configuration
└── ml_models/
    └── saved_models/     # Trained ML models storage
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Supabase account
- Google OAuth credentials (optional)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd agrismart-backend
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Update the following variables:

```env
# Supabase Configuration
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# JWT Configuration  
SECRET_KEY=your_super_secret_jwt_key_here_make_it_long
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Google OAuth (Optional)
GOOGLE_CLIENT_ID=your_google_oauth_client_id
GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret

# Frontend API Key (Share with frontend teams)
API_KEY=AgriSmart-2025-Key123
```

### 4. Database Setup (Supabase)

Execute the following SQL in your Supabase SQL editor:

```sql
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    phone VARCHAR(15) NOT NULL,
    region VARCHAR(100) NOT NULL,
    farm_size NUMERIC(10, 2) NOT NULL,
    main_crops VARCHAR(200) NOT NULL,
    password_hash TEXT NOT NULL,
    member_since INTEGER NOT NULL,
    predictions_count INTEGER DEFAULT 0,
    accuracy_rate VARCHAR(10) DEFAULT '0%',
    last_prediction VARCHAR(50) DEFAULT 'Never',
    created_at TIMESTAMP DEFAULT now()
);

-- Predictions table
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    prediction_type VARCHAR NOT NULL,
    crop_type VARCHAR NOT NULL,
    input_data JSONB NOT NULL,
    predictions JSONB NOT NULL,
    confidence FLOAT NOT NULL,
    recommendations JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

-- Irrigation logs table
CREATE TABLE irrigation_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    schedule_date TIMESTAMP NOT NULL,
    duration_minutes INTEGER NOT NULL,
    water_volume NUMERIC(10, 2) NOT NULL,
    weather_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE irrigation_logs ENABLE ROW LEVEL SECURITY;

-- RLS Policies for team collaboration
CREATE POLICY "Users can view own data" ON users FOR SELECT USING (auth.uid()::text = id::text);
CREATE POLICY "Users can update own data" ON users FOR UPDATE USING (auth.uid()::text = id::text);
CREATE POLICY "Users can view own predictions" ON predictions FOR SELECT USING (auth.uid()::text = user_id::text);
CREATE POLICY "Users can create predictions" ON predictions FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);
CREATE POLICY "Users can view own irrigation logs" ON irrigation_logs FOR SELECT USING (auth.uid()::text = user_id::text);
CREATE POLICY "Users can create irrigation logs" ON irrigation_logs FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);
```

### 5. Run the Application

#### Development
```bash
s


```

#### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 6. Verify Installation

- API Documentation: http://localhost:8000/api/docs
- Health Check: http://localhost:8000/api/health
- Root Endpoint: http://localhost:8000/

## 📡 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - User login  
- `POST /api/auth/google` - Google OAuth login
- `GET /api/auth/me` - Get current user info
- `POST /api/auth/refresh` - Refresh JWT token

### ML Predictions  
- `POST /api/predictions/yield` - Predict crop yield
- `POST /api/predictions/disease` - Detect crop disease
- `POST /api/predictions/pest` - Classify crop pests
- `GET /api/predictions/history` - Get prediction history

### Irrigation Management
- `POST /api/irrigation/schedule` - Calculate irrigation schedule
- `GET /api/irrigation/history` - Get irrigation history
- `GET /api/irrigation/recommendations` - Get irrigation tips
- `GET /api/irrigation/efficiency` - Analyze water efficiency

### Dashboard & Analytics
- `GET /api/dashboard/stats` - Get dashboard statistics
- `GET /api/dashboard/analytics` - Get crop analytics
- `GET /api/dashboard/overview` - Get comprehensive overview  
- `GET /api/dashboard/insights` - Get AI-powered insights

## 🔧 Frontend Integration

### CORS Configuration
The backend is configured to accept requests from:
- `https://agrismart-phi.vercel.app` (Production)
- `http://localhost:3000` (Development)

### API Key Authentication
For frontend teams, use the API key in headers:
```javascript
const API_KEY = "AgriSmart-2025-Key123";

// Example API call
const response = await fetch('http://localhost:8000/api/auth/login', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY  // Include for protected routes
  },
  body: JSON.stringify({
    email: "farmer@example.com",
    password: "securePassword123"
  })
});
```

### JWT Authentication
For authenticated endpoints, include the JWT token:
```javascript
const token = "your_jwt_token_here";

const response = await fetch('http://localhost:8000/api/predictions/yield', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  },
  body: JSON.stringify({
    prediction_type: "yield",
    crop_type: "wheat",
    area: 2.5,
    soil_ph: 6.8,
    nitrogen: 45,
    phosphorus: 25,
    potassium: 35,
    rainfall: 120,
    temperature: 28,
    sowing_date: "2025-03-15"
  })
});
```

## 🤖 Machine Learning Models

### Yield Prediction
- **Algorithm**: Random Forest Regressor
- **Features**: Soil pH, NPK values, rainfall, temperature
- **Accuracy**: ~85% on test data
- **Output**: Estimated yield per hectare, quality grade

### Disease Detection  
- **Type**: Mock classification (expandable)
- **Input**: Symptoms, affected area, duration
- **Output**: Disease type, severity, treatment recommendations

### Pest Classification
- **Type**: Mock classification (expandable)  
- **Input**: Pest description, damage level, treatment history
- **Output**: Pest type, damage score, control recommendations

## 💧 Irrigation Intelligence

### Scheduling Algorithm
The irrigation service considers:
- **Soil moisture levels** (critical: <20%, optimal: 60%)
- **Weather conditions** (rainfall, temperature, humidity)
- **Crop water requirements** (crop-specific needs)
- **Seasonal adjustments** (growth stage considerations)

### Smart Recommendations
- Optimal timing (early morning/evening)
- Water volume calculations
- Efficiency improvements
- Weather-based adjustments

## 🚀 Deployment

### Render.com Deployment

1. **Connect Repository**: Link your GitHub repository to Render

2. **Environment Variables**: Set in Render dashboard:
```
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
SECRET_KEY=your_secret_key
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
API_KEY=AgriSmart-2025-Key123
```

3. **Build Command**: 
```bash
pip install -r requirements.txt
```

4. **Start Command**:
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Alternative Deployment Options

#### Docker
```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Railway
```bash
railway login
railway init
railway add
railway up
```

## 👥 Team Collaboration

### For the 4 Teams Working on SIH25044

#### Team Roles
1. **Frontend Team**: React dashboard integration
2. **ML Team**: Model optimization and training  
3. **Mobile Team**: Mobile app API integration
4. **DevOps Team**: Deployment and monitoring

#### Development Workflow
1. **Branch Strategy**: feature/team-name/feature-description
2. **API Testing**: Use `/api/docs` for interactive testing
3. **Environment**: Shared staging environment on Render
4. **Communication**: API changes communicated via team chat

#### Shared Resources
- **API Documentation**: Always available at `/api/docs`
- **API Key**: `AgriSmart-2025-Key123` (shared with all teams)
- **Test Database**: Shared Supabase instance for development
- **Monitoring**: Logs available via Render dashboard

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/ -v
```

### API Testing Examples
```python
import httpx
import pytest

@pytest.mark.asyncio
async def test_user_registration():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/auth/register",
            json={
                "name": "Test Farmer",
                "email": "test@farmer.com",
                "phone": "+91-9876543210",
                "region": "Punjab",
                "farm_size": 5.0,
                "main_crops": "wheat, rice",
                "password": "SecurePass123!"
            }
        )
        assert response.status_code == 201
```

## 🔍 Monitoring & Logging

### Log Levels
- **INFO**: Normal operations, user actions
- **WARNING**: Non-critical issues, validation failures  
- **ERROR**: Application errors, failed operations
- **DEBUG**: Detailed debugging information

### Key Metrics to Monitor
- API response times
- Authentication success rates  
- ML prediction accuracy
- Database query performance
- Error rates by endpoint

## 🛡️ Security Features

### Authentication & Authorization
- JWT token-based authentication
- Google OAuth integration  
- Password hashing with bcrypt
- Row Level Security (RLS) in Supabase

### Data Protection
- Input validation with Pydantic
- SQL injection prevention
- CORS configuration
- Rate limiting (basic implementation)

### Privacy Compliance
- User data encryption at rest
- Minimal data collection
- Secure credential storage
- API key management

## 📊 Performance Optimization

### Database Optimization
- Efficient queries with proper indexing
- Connection pooling via Supabase
- Async operations throughout
- Minimal data transfer

### API Performance  
- FastAPI async framework
- Response caching for static data
- Pagination for large datasets
- Background tasks for ML operations

## 🐛 Troubleshooting

### Common Issues

#### 1. Database Connection Failed
```bash
# Check environment variables
echo $SUPABASE_URL
echo $SUPABASE_ANON_KEY

# Verify Supabase connection
curl -H "apikey: $SUPABASE_ANON_KEY" $SUPABASE_URL/rest/v1/
```

#### 2. JWT Token Invalid
```bash
# Check SECRET_KEY is set
echo $SECRET_KEY

# Verify token in JWT debugger: jwt.io
```

#### 3. CORS Errors
- Ensure frontend URL is in CORS origins
- Check request headers include proper Content-Type
- Verify OPTIONS requests are handled

#### 4. ML Model Loading Failed
```bash
# Check models directory exists
ls -la app/ml_models/saved_models/

# Recreate models directory
mkdir -p app/ml_models/saved_models
```

### Debug Mode
```bash
# Run with debug logging
DEBUG=true uvicorn app.main:app --reload --log-level debug
```

## 📞 Support & Contact

### For Team Members
- **Technical Issues**: Create GitHub issues with detailed logs
- **API Questions**: Check `/api/docs` first, then ask in team chat
- **Database Issues**: Contact DevOps team lead
- **Deployment Problems**: Check Render logs and environment variables

### Documentation
- **API Docs**: http://localhost:8000/api/docs (auto-generated)
- **ReDoc**: http://localhost:8000/api/redoc (alternative format)
- **OpenAPI Schema**: http://localhost:8000/api/openapi.json

## 🎯 Roadmap & Future Enhancements

### Phase 1 (Current - SIH Submission)
- ✅ Core API endpoints
- ✅ Basic ML models
- ✅ Authentication system
- ✅ Dashboard analytics

### Phase 2 (Post-SIH)
- 🔄 Advanced ML model training
- 🔄 Real weather API integration  
- 🔄 Mobile push notifications
- 🔄 Advanced analytics dashboard

### Phase 3 (Future)
- 📱 IoT sensor integration
- 🛰️ Satellite imagery analysis
- 🤖 Advanced AI recommendations
- 📊 Marketplace integration

## 🏆 Smart India Hackathon 2025

**Problem Statement**: SIH25044 - Smart Irrigation and Crop Monitoring System

**Team Goal**: Develop an AI-powered solution for optimizing irrigation schedules and monitoring crop health to help Indian farmers increase productivity while conserving water resources.

**Impact**: Targeting 10,000+ small and medium farmers in the first year, with potential for 25% water savings and 15% yield improvement.

---

## 📄 License

This project is developed for Smart India Hackathon 2025. All rights reserved to the development team.

## 🙏 Acknowledgments

- Smart India Hackathon 2025 organizers
- Supabase for database infrastructure  
- FastAPI community for excellent documentation
- All team members contributing to this project

---

**Happy Farming! 🌾** 

For any issues or questions, please refer to the API documentation at `/api/docs` or contact the development team.