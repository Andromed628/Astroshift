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

# Load SPHEREx facts context dynamically if available
FACTS_PATH = os.path.join(os.path.dirname(__file__), "spherex_facts.md")
SPHEREX_KNOWLEDGE = ""
if os.path.exists(FACTS_PATH):
    try:
        with open(FACTS_PATH, "r", encoding="utf-8") as f:
            SPHEREX_KNOWLEDGE = f.read()
    except Exception:
        pass

SYSTEM_PROMPT = f"""
You are AstroBot, an aerospace-grade AI co-pilot for AstroShift (NASA Space Apps Challenge 2026).

Domain Grounding Knowledge:
{SPHEREX_KNOWLEDGE}

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
    history_list = history or []
    context_str = f"\nActive Frame Context: {dataset_context}" if dataset_context else ""
    
    # Read keys dynamically on every incoming request
    groq_key = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    nvidia_key = os.getenv("NVIDIA_API_KEY")

    error_logs = []

    # --------------------------------------------------
    # 1. MULTIMODAL QUERY (If Image Provided)
    # --------------------------------------------------
    if image_base64:
        clean_b64 = image_base64.split(",")[-1] if "," in image_base64 else image_base64
        
        # Tier 1 Multimodal: Gemini 2.0 Flash
        if gemini_key:
            genai.configure(api_key=gemini_key)
            try:
                img_bytes = base64.b64decode(clean_b64)
                pil_img = Image.open(io.BytesIO(img_bytes))
                prompt_content = f"{SYSTEM_PROMPT}{context_str}\n\nUser Question:\n{message}"
                
                for gemini_model in ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash-latest"]:
                    try:
                        model = genai.GenerativeModel(gemini_model)
                        res = await model.generate_content_async([prompt_content, pil_img])
                        if res and res.text and res.text.strip():
                            return res.text.strip()
                    except Exception as e:
                        error_logs.append(f"Gemini Vision ({gemini_model}): {str(e)}")
            except Exception as e:
                error_logs.append(f"Gemini Vision processing error: {str(e)}")

        # Tier 2 Multimodal: NVIDIA Vision
        if nvidia_key:
            try:
                nvidia_client = AsyncOpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=nvidia_key
                )
                for nv_vision in ["nvidia/neva-22b", "nvidia/nemotron-3.5-super-vl-120b-a12b"]:
                    try:
                        res = await nvidia_client.chat.completions.create(
                            model=nv_vision,
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
                        error_logs.append(f"NVIDIA Vision ({nv_vision}): {str(e)}")
            except Exception as e:
                error_logs.append(f"NVIDIA Vision client error: {str(e)}")

    # --------------------------------------------------
    # 2. TEXT QUERY / FAILOVER CHAIN
    # --------------------------------------------------
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}{context_str}"}]
    for msg in history_list[-6:]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
    messages.append({"role": "user", "content": message})

    # Tier 1 Text: Groq (Llama 3.1 8B Instant / 70B Active Models)
    if groq_key:
        try:
            groq_client = AsyncGroq(api_key=groq_key)
            for groq_model in ["llama-3.1-8b-instant", "llama3-70b-8192", "llama-3.3-70b-specdec"]:
                try:
                    completion = await groq_client.chat.completions.create(
                        model=groq_model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=512
                    )
                    reply = completion.choices[0].message.content
                    if reply and reply.strip():
                        return reply.strip()
                except Exception as e:
                    error_logs.append(f"Groq ({groq_model}): {str(e)}")
        except Exception as e:
            error_logs.append(f"Groq Client Error: {str(e)}")

    # Tier 2 Text: NVIDIA NIM (Llama 3.3 70B / Nemotron)
    if nvidia_key:
        try:
            nvidia_client = AsyncOpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=nvidia_key
            )
            for nvidia_model in ["meta/llama-3.3-70b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct"]:
                try:
                    res = await nvidia_client.chat.completions.create(
                        model=nvidia_model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=512
                    )
                    reply = res.choices[0].message.content
                    if reply and reply.strip():
                        return reply.strip()
                except Exception as e:
                    error_logs.append(f"NVIDIA NIM ({nvidia_model}): {str(e)}")
        except Exception as e:
            error_logs.append(f"NVIDIA Client Error: {str(e)}")

    # Tier 3 Text: Gemini 2.0 Flash
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            full_text_prompt = f"{SYSTEM_PROMPT}{context_str}\n\nUser Question:\n{message}"
            for gemini_model in ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash-latest"]:
                try:
                    model = genai.GenerativeModel(gemini_model)
                    res = await model.generate_content_async(full_text_prompt)
                    if res and res.text and res.text.strip():
                        return res.text.strip()
                except Exception as e:
                    error_logs.append(f"Gemini ({gemini_model}): {str(e)}")
        except Exception as e:
            error_logs.append(f"Gemini Client Error: {str(e)}")

    diag_info = " | ".join(error_logs) if error_logs else "No active provider keys found in environment"
    raise RuntimeError(f"AstroBot telemetry failed. Provider Errors -> {diag_info}")

generate_astrobot_reply = ask_astrobot
