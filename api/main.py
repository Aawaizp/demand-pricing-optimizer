# api/main.py
# FastAPI application for demand forecasting and price optimization

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="AI-Powered Demand Forecasting & Price Optimization API",
    description="Predict product demand and optimize pricing for maximum revenue using ML",
    version="1.0.0"
)

# Enable CORS (so frontend can call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routes
from api.routes import router
app.include_router(router)

# Root endpoint
@app.get("/")
def root():
    """
    Welcome endpoint - shows API info
    """
    return {
        "message": "Welcome to AI Demand Forecasting & Price Optimization API!",
        "version": "1.0.0",
        "endpoints": {
            "predict_demand": "/predict-demand",
            "suggest_price": "/suggest-price",
            "health": "/health"
        },
        "documentation": "/docs"
    }

# Health check endpoint
@app.get("/health")
def health_check():
    """
    Check if API is running
    """
    return {
        "status": "healthy",
        "message": "API is running successfully!"
    }

# Run with: uvicorn api.main:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)