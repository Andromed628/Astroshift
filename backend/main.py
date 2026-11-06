import os
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from services.astrobot import ask_astrobot
from services.astronomical_engine import (
    parse_fits_header,
    align_images,
    resolve_celestial_target,
    numpy_to_base64_png
)
from services.analytics import analyze_delta

load_dotenv()

app = FastAPI(
    title="AstroShift Backend API",
    description="SPHEREx Astronomical Processing Engine & AstroBot AI Telemetry",
    version="2.0.0"
)

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
origins = [origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

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

class DeltaAnalysisRequest(BaseModel):
    matrix_a: List[List[float]]
    matrix_b: List[List[float]]
    sigma: Optional[float] = 3.0
    include_plots: Optional[bool] = True

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "system": "AstroShift Core Engine V2",
        "environment": "Render Production"
    }

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

@app.get("/api/search")
async def search_celestial_target(query: str = Query(..., description="Target name, e.g., Messier 31")):
    try:
        result = resolve_celestial_target(query)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/blink-data")
async def process_blink_data(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...)
):
    try:
        bytes_a = await file_a.read()
        bytes_b = await file_b.read()

        data_a, meta_a = parse_fits_header(bytes_a)
        data_b, meta_b = parse_fits_header(bytes_b)

        if data_a is None or data_b is None:
            raise HTTPException(status_code=400, detail="Invalid FITS file structure.")

        aligned_b, _ = await asyncio.to_thread(align_images, data_b, data_a)

        b64_a = f"data:image/png;base64,{numpy_to_base64_png(data_a)}"
        b64_b = f"data:image/png;base64,{numpy_to_base64_png(aligned_b)}"

        return {
            "status": "success",
            "dataset_id": "spherex-fits-live",
            "image_a": b64_a,
            "image_b": b64_b,
            "frame_a_b64": b64_a,
            "frame_b_aligned_b64": b64_b,
            "metadata_a": meta_a,
            "metadata_b": meta_b
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blink processing error: {str(e)}")

@app.post("/api/analytics/delta")
async def compute_delta_analysis(request: DeltaAnalysisRequest):
    try:
        result = await asyncio.to_thread(
            analyze_delta,
            matrix_a=request.matrix_a,
            matrix_b=request.matrix_b,
            sigma=request.sigma,
            include_plots=request.include_plots
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delta processing failure: {str(e)}")
