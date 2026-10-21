import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from services.astrobot import generate_astrobot_reply

load_dotenv()

app = FastAPI(
    title="AstroShift Backend API",
    version="1.0.0",
    description="SPHEREx Dataset Processing & AstroBot API"
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class BlinkRequest(BaseModel):
    dataset_id: str = "spherex-01"

class DeltaAnomaly(BaseModel):
    id: int
    x: int
    y: int
    flux_change_pct: float
    type: str

class BlinkResponse(BaseModel):
    dataset_id: str
    target_name: str
    ra_deg: float
    dec_deg: float
    image_a: str
    image_b: str
    delta_anomalies: List[DeltaAnomaly]

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    image_base64: Optional[str] = None
    dataset_context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    reply: str

# Endpoints
@app.get("/")
def read_root():
    return {"system": "AstroShift Backend API", "status": "active"}

@app.get("/api/health")
def health_check():
    return {"status": "healthy", "service": "AstroShift API"}

@app.post("/api/blink-data", response_model=BlinkResponse)
def get_blink_data(payload: BlinkRequest):
    mock_path = os.path.join(os.path.dirname(__file__), "data", "mockData.json")
    
    if os.path.exists(mock_path):
        with open(mock_path, "r") as f:
            data = json.load(f)
            dataset = data.get(payload.dataset_id, data.get("spherex-01"))
            return BlinkResponse(**dataset)
            
    return BlinkResponse(
        dataset_id=payload.dataset_id,
        target_name="SPHEREx Deep Field Alpha",
        ra_deg=184.56,
        dec_deg=-45.12,
        image_a="https://via.placeholder.com/512?text=SPHEREx+Frame+A",
        image_b="https://via.placeholder.com/512?text=SPHEREx+Frame+B",
        delta_anomalies=[
            DeltaAnomaly(id=1, x=120, y=340, flux_change_pct=14.2, type="Transient / Asteroid")
        ]
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    try:
        reply = await generate_astrobot_reply(
            message=payload.message,
            history=[msg.model_dump() for msg in payload.history],
            image_base64=payload.image_base64,
            dataset_context=payload.dataset_context
        )
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
