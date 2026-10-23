import os
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from groq import AsyncGroq
from openai import AsyncOpenAI
from dotenv import load_dotenv
from .astrobot_science import generate_scientific_reply_async

load_dotenv()

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", "mock_key"))

if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

nvidia_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY", "mock_key")
)

SYSTEM_PROMPT = """
You are AstroBot, an aerospace-grade AI co-pilot for AstroShift (NASA Space Apps Challenge 2026).
Your domain expertise:
1. NASA SPHEREx infrared sky survey dataset analysis and broadband flux metrics.
2. Celestial coordinate transformations (Right Ascension, Declination, WCS mapping).
3. Image registration, star/asteroid alignment, and frame differencing anomalies.

Keep responses precise, scientific, concise, and structured for researchers and judges.
"""

async def generate_astrobot_reply(
    message: str,
    history: List[Dict[str, str]],
    image_base64: Optional[str] = None,
    dataset_context: Optional[Dict[str, Any]] = None
) -> str:
    
    # 1. VISUAL QUERY: Gemini 1.5 Flash -> NVIDIA NIM Fallback
    if image_base64:
        try:
            model = genai.GenerativeModel("gemini-3.6-flash")

            res = await model.generate_content_async([
                SYSTEM_PROMPT,
                {"mime_type": "image/png", "data": image_base64},
                f"Context: {dataset_context or {}}\nUser Question: {message}"
            ])
            return res.text
        except Exception as e:
            print(f"[AstroBot] Gemini failed: {e}. Falling back to NVIDIA NIM...")

        try:
            res = await nvidia_client.chat.completions.create(
                model="nvidia/nemotron-3.5-super-vl-120b-a12b",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{SYSTEM_PROMPT}\n{message}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }],
                max_tokens=512
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"[AstroBot] NVIDIA Vision failed: {e}. Falling back to Groq Text...")
    # 2. TEXT QUERY: AstroBot V2 with Gemini Fallback
    try:
        reply = await generate_scientific_reply_async(
            message=message,
            history=history,
            dataset_context=dataset_context
        )
        return reply

    except Exception as e:
        print(f"[AstroBot] Groq V2 rate-limited/failed: {e}. Falling back to Gemini...")
        try:
            gemini_model = genai.GenerativeModel("gemini-3.6-flash")
            fallback_reply = gemini_model.generate_content(
                f"Act as a NASA SPHEREx scientist. Answer this concisely: {message}"
            )
            return fallback_reply.text
        except Exception as gemini_error:
            print(f"[AstroBot] Gemini fallback also failed: {gemini_error}")
            return "AstroBot telemetry offline. Please check server logs."


