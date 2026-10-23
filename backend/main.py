import os
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Internal backend service imports
from services.astrobot import ask_astrobot
from services.astronomical_engine import (
    parse_fits_header,
    align_images,
    resolve_celestial_target,
    numpy_to_base64_png
)

app = FastAPI(
    title="AstroShift Backend API",
    description="SPHEREx Astronomical Processing Engine & AstroBot AI Telemetry",
    version="2.0.0"
)

# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# REQUEST & RESPONSE SCHEMAS
# --------------------------------------------------
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


# --------------------------------------------------
# ROUTE 1: HEALTH CHECK
# --------------------------------------------------
@app.get("/")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "system": "AstroShift Core Engine V2",
        "environment": "Render Production"
    }


# --------------------------------------------------
# ROUTE 2: ASTROBOT AI CHAT (3-TIER FAILOVER)
# --------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    if not payload.message or not payload.message.strip():
        raise HTTPException(status_code=400, detail="Chat message cannot be empty.")
    
    try:
        reply = await ask_astrobot(
            message=payload.message,
            history=[msg.model_dump() for msg in payload.history] if payload.history else None,
            dataset_context=payload.dataset_context,
            image_base64=payload.image_base64
        )
        return ChatResponse(reply=reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------
# ROUTE 3: CELESTIAL TARGET SEARCH RESOLVER
# --------------------------------------------------
@app.get("/api/search")
async def search_celestial_target(query: str = Query(..., description="Target name, e.g., Messier 31")):
    try:
        result = resolve_celestial_target(query)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# --------------------------------------------------
# ROUTE 4: ASTRONOMICAL FITS BLINK & ALIGNMENT DATA
# --------------------------------------------------
@app.post("/api/blink-data")
async def process_blink_data(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...)
):
    """
    Receives two FITS files (Frame A and Frame B), parses headers,
    aligns Frame B to Frame A using point-source triangulation,
    and returns Base64 encoded PNG strings for frontend rendering.
    """
    try:
        bytes_a = await file_a.read()
        bytes_b = await file_b.read()

        # Parse FITS files
        data_a, meta_a = parse_fits_header(bytes_a)
        data_b, meta_b = parse_fits_header(bytes_b)

        if data_a is None or data_b is None:
            raise HTTPException(status_code=400, detail="Invalid FITS file structure.")

        # Non-blocking heavy star registration in worker thread
        aligned_b, _ = await asyncio.to_thread(align_images, data_b, data_a)

        # Convert arrays to Base64 PNGs
        b64_frame_a = numpy_to_base64_png(data_a)
        b64_frame_b_aligned = numpy_to_base64_png(aligned_b)

        return {
            "status": "success",
            "metadata_a": meta_a,
            "metadata_b": meta_b,
            "frame_a_b64": b64_frame_a,
            "frame_b_aligned_b64": b64_frame_b_aligned
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blink processing error: {str(e)}")
