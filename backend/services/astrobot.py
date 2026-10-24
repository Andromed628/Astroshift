import os
import io
import base64
from typing import Optional, List, Dict, Any
import google.generativeai as genai
from groq import AsyncGroq
from openai import AsyncOpenAI
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Initialize API clients safely
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None

if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

nvidia_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
) if os.getenv("NVIDIA_API_KEY") else None

SYSTEM_PROMPT = """
You are AstroBot, an aerospace-grade AI co-pilot for AstroShift (NASA Space Apps Challenge 2026).
Your domain expertise:
1. NASA SPHEREx infrared sky survey dataset analysis and broadband flux metrics.
2. Celestial coordinate transformations (Right Ascension, Declination, WCS mapping).
3. Image registration, star/asteroid alignment, and frame differencing anomalies.

Keep responses precise, scientific, concise, and structured for researchers and judges.
"""

async def ask_astrobot(
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    image_base64: Optional[str] = None,
    dataset_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Primary AstroBot entry point with 3-tier resilience:
    Groq (Primary) -> NVIDIA NIM (Secondary) -> Gemini 1.5 Flash (Tertiary / Multimodal)
    """
    history_list = history or []
    context_str = f"\nActive Frame Context: {dataset_context}" if dataset_context else ""
    
    # --------------------------------------------------
    # 1. MULTIMODAL QUERY (If Image Provided)
    # --------------------------------------------------
    if image_base64:
        clean_b64 = image_base64.split(",")[-1] if "," in image_base64 else image_base64
        
        # Try Gemini Flash Multimodal First
        if os.getenv("GEMINI_API_KEY"):
            try:
                img_bytes = base64.b64decode(clean_b64)
                pil_img = Image.open(io.BytesIO(img_bytes))
                model = genai.GenerativeModel("gemini-1.5-flash")
                
                prompt_content = f"{SYSTEM_PROMPT}{context_str}\n\nUser Question:\n{message}"
                res = await model.generate_content_async([prompt_content, pil_img])
                if res and res.text and res.text.strip():
                    return res.text.strip()
            except Exception as e:
                print(f"[AstroBot] Gemini Vision failed: {e}. Trying NVIDIA NIM Vision...")

        # Fallback to NVIDIA Vision
        if nvidia_client:
            try:
                res = await nvidia_client.chat.completions.create(
                    model="nvidia/nemotron-3.5-super-vl-120b-a12b",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{SYSTEM_PROMPT}{context_str}\n\n{message}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{clean_b64}"}}
                        ]
                    }],
                    max_tokens=512
                )
                reply = res.choices[0].message.content
                if reply and reply.strip():
                    return reply.strip()
            except Exception as e:
                print(f"[AstroBot] NVIDIA Vision failed: {e}. Proceeding to text failover...")

    # --------------------------------------------------
    # 2. TEXT QUERY / FAILOVER CHAIN
    # --------------------------------------------------
    
    # Format messages array including history
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}{context_str}"}]
    for msg in history_list[-6:]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": message})

    # Tier 1: Groq
    if groq_client:
        try:
            completion = await groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.3,
                max_tokens=512
            )
            reply = completion.choices[0].message.content
            if reply and reply.strip():
                return reply.strip()
        except Exception as e:
            print(f"[AstroBot] Groq Text failed: {e}. Escalating to NVIDIA NIM...")

    # Tier 2: NVIDIA NIM Text
    if nvidia_client:
        try:
            res = await nvidia_client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=messages,
                temperature=0.3,
                max_tokens=512
            )
            reply = res.choices[0].message.content
            if reply and reply.strip():
                return reply.strip()
        except Exception as e:
            print(f"[AstroBot] NVIDIA NIM Text failed: {e}. Escalating to Gemini...")

    # Tier 3: Gemini Text
    if os.getenv("GEMINI_API_KEY"):
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            full_text_prompt = f"{SYSTEM_PROMPT}{context_str}\n\nUser Question:\n{message}"
            res = await model.generate_content_async(full_text_prompt)
            if res and res.text and res.text.strip():
                return res.text.strip()
        except Exception as e:
            print(f"[AstroBot] Gemini Text failed: {e}")

    raise RuntimeError("AstroBot scientific telemetry is currently unavailable across all AI model clusters.")

# Backward compatibility alias
generate_astrobot_reply = ask_astrobot
