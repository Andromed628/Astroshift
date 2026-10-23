import os
import google.generativeai as genai
from openai import AsyncOpenAI
from .astrobot_science import generate_scientific_reply_async, SYSTEM_CONTEXT

# Configure Gemini fallback
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# Configure NVIDIA NIM fallback (uses OpenAI SDK with NVIDIA endpoint)
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY")
nvidia_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_KEY
) if NVIDIA_KEY else None


async def ask_astrobot(message: str, history: list = None, dataset_context: dict = None):
    """
    Primary entry point for AstroBot V2 interface.
    
    3-Tier Resilience Architecture:
    1. Groq (Primary: Qwen 27B / Llama 3)
    2. NVIDIA NIM (Secondary: Llama 3.1 70B Instruct)
    3. Gemini (Tertiary: Gemini 1.5 Flash)
    """
    # --------------------------------------------------
    # TIER 1: Primary Groq Scientific Engine
    # --------------------------------------------------
    try:
        return await generate_scientific_reply_async(
            message=message,
            history=history,
            dataset_context=dataset_context
        )
    except Exception as groq_error:
        print(f"[AstroBot] Groq primary failed: {groq_error}. Escalating to NVIDIA NIM...")

    # --------------------------------------------------
    # TIER 2: Secondary NVIDIA NIM Fallback
    # --------------------------------------------------
    if nvidia_client:
        try:
            response = await nvidia_client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[
                    {"role": "system", "content": SYSTEM_CONTEXT},
                    {"role": "user", "content": message}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            reply = response.choices[0].message.content
            if reply and reply.strip():
                return reply.strip()
        except Exception as nvidia_error:
            print(f"[AstroBot] NVIDIA NIM fallback failed: {nvidia_error}. Escalating to Gemini...")

    # --------------------------------------------------
    # TIER 3: Tertiary Gemini 1.5 Flash Fallback
    # --------------------------------------------------
    if GEMINI_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt_payload = f"{SYSTEM_CONTEXT}\n\nUser Question:\n{message}"

            response = await model.generate_content_async(prompt_payload)
            if response and response.text and response.text.strip():
                return response.text.strip()
        except Exception as gemini_error:
            print(f"[AstroBot] Gemini fallback failed: {gemini_error}")

    raise RuntimeError(
        "AstroBot scientific telemetry is currently unavailable across all AI model clusters."
    )

