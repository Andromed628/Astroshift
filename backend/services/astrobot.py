import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import AsyncGroq
import google.generativeai as genai
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("astrobot")

app = FastAPI(title="AstroShift AstroBot API")

# Setup CORS using Render environment variable
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys from Render Environment Variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# 1. Initialize Groq Client for Qwen Text/Science Model
groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
QWEN_MODEL = "qwen/qwen3.8-27b"

# 2. Configure Gemini Vision
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# 3. Initialize NVIDIA Async Client for Vision Fallback
nvidia_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
) if NVIDIA_API_KEY else None

# Prompts
SPHEREX_SYSTEM_PROMPT = """
You are AstroBot, an expert astronomical AI assistant for the AstroShift platform.
Specializing in infrared astronomy, SPHEREx payload science, and celestial anomaly identification.
Provide rigorous, science-grounded explanations incorporating observational dataset context when provided.
"""

# Request Schemas
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class TextChatRequest(BaseModel):
    messages: List[ChatMessage]
    dataset_context: Optional[str] = None

class VisionChatRequest(BaseModel):
    prompt: str
    image_base64: str


@app.get("/api/health")
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "AstroBot V2"}

# =====================================================================
# Text & Science Generation Route (Qwen via Groq)
# =====================================================================
@app.post("/api/chat/science")
async def chat_science_text(request: TextChatRequest):
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on server.")

    try:
        # Build prompt payload with SPHEREx context
        system_content = SPHEREX_SYSTEM_PROMPT
        if request.dataset_context:
            system_content += f"\n\n[Active Astronomical Dataset Context]\n{request.dataset_context}"

        messages_payload = [{"role": "system", "content": system_content}]
        messages_payload.extend([{"role": m.role, "content": m.content} for m in request.messages])

        # Execute async call to Groq Qwen
        response = await groq_client.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages_payload,
            temperature=0.3,
            max_tokens=2048
        )

        reply_text = response.choices[0].message.content
        return {
            "role": "assistant",
            "content": reply_text,
            "provider": "GroqCloud",
            "model": QWEN_MODEL
        }

    except Exception as e:
        logger.error(f"Groq Qwen execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text generation error: {str(e)}")

# =====================================================================
# Vision Route (Gemini Vision with NVIDIA Fallback)
# =====================================================================
@app.post("/api/chat/vision")
async def chat_vision_analysis(request: VisionChatRequest):
    # Attempt 1: Gemini Vision Primary Route
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            image_data = {"mime_type": "image/jpeg", "data": request.image_base64}
            response = model.generate_content([request.prompt, image_data])
            return {
                "role": "assistant",
                "content": response.text,
                "provider": "Gemini Vision"
            }
        except Exception as e:
            logger.warning(f"Gemini Vision failed, attempting NVIDIA fallback: {str(e)}")

    # Attempt 2: NVIDIA Vision Fallback Route
    if nvidia_client:
        try:
            response = await nvidia_client.chat.completions.create(
                model="nvidia/neva-22b",
                messages=[
                    {
                        "role": "user",
                        "content": f'{request.prompt} <img src="data:image/jpeg;base64,{request.image_base64}" />'
                    }
                ],
                max_tokens=1024
            )
            return {
                "role": "assistant",
                "content": response.choices[0].message.content,
                "provider": "NVIDIA Vision Fallback"
            }
        except Exception as e:
            logger.error(f"NVIDIA Vision Fallback also failed: {str(e)}")

    raise HTTPException(status_code=500, detail="Both Gemini Vision and NVIDIA Vision fallback failed.")
