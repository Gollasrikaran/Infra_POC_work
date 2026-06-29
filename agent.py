import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import base64

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PRIMARY_MODEL  = os.getenv("GEMINI_MODEL")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = os.getenv("GROQ_MODEL")


PROMPT = """You are an expert highway earthwork engineer and cross-section analyst.
Analyze the provided cross-section image and focus ONLY on visual interpretation and region classification.

IMPORTANT RULES:

1. Identify the Existing Ground profile.
   - Usually represented by a dashed or dotted line.
   - Follow the profile continuously across the cross-section.
   - Trace its path and return it as a list of coordinates.

2. Identify the Proposed Grade profile.
   - Usually represented by a solid line.
   - Follow the profile continuously across the cross-section.
   - Trace its path and return it as a list of coordinates.

3. Determine where the Proposed Grade lies above or below the Existing Ground.

4. Classify regions:
   - FILL: Proposed Grade is ABOVE Existing Ground. Color interpretation = GREEN.
   - CUT: Existing Ground is ABOVE Proposed Grade. Color interpretation = RED.

5. Focus heavily on image understanding.

6. Ignore:
   - Area calculations
   - Volume calculations
   - Quantity estimates
   - Engineering reports

7. Describe exactly which portions of the image should be visually colored.

8. If uncertain, explicitly state the uncertainty rather than guessing.

9. Analyze:
    - Profile continuity
    - Line style differences
    - Relative elevations
    - Catch point locations
    - Crossing points

10. The goal is visual validation of cut/fill regions, not numerical computation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COORDINATE SPECIFICATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trace the profiles using normalized coordinates:
- X range is 0 to 1000 (0 is left edge of image, 1000 is right edge of image).
- Y range is 0 to 1000 (0 is top edge of image, 1000 is bottom edge of image).
- Sample each profile at regular horizontal intervals (every 20 to 50 units in X range, or at key vertices/inflection points) to accurately capture its shape. Return at least 15-20 points for each profile if visible.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — Respond ONLY with valid JSON, no markdown, no code fences, no extra text:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "station": "station number if visible, else null",
  "existing_ground_profile": [
    {"x": <integer 0-1000>, "y": <integer 0-1000>}
  ],
  "proposed_grade_profile": [
    {"x": <integer 0-1000>, "y": <integer 0-1000>}
  ],
  "regions": [
    {
      "region_id": <integer>,
      "classification": "CUT" or "FILL",
      "color": "RED" or "GREEN",
      "confidence": <integer 0-100>,
      "reasoning": "Explain why this region is classified as CUT or FILL based on profile heights."
    }
  ],
  "visual_coloring_recommendations": {
    "green_regions": "detailed description of where green color (FILL) should appear in the image",
    "red_regions": "detailed description of where red color (CUT) should appear in the image",
    "why_selected": "detailed explanation of why these regions were selected for coloring"
  },
  "uncertainties": "explicit description of any uncertainties, or null if none"
}"""


def _extract_image_text(img_bytes: bytes, filename: str) -> str:
    """Send image to Groq (if configured) or Gemini, and return raw JSON string."""
    image_base64 = base64.b64encode(img_bytes).decode("utf-8")
    
    if GROQ_API_KEY:
        import requests
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL or "qwen/qwen3-32b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result_json = response.json()
            return result_json["choices"][0]["message"]["content"]
        except Exception as e:
            if 'response' in locals() and hasattr(response, 'text'):
                return f"Error calling Groq API: {str(e)} - Response: {response.text}"
            return f"Error calling Groq API: {str(e)}"
            
    else:
        # Fallback to Gemini
        llm = ChatGoogleGenerativeAI(
            model=PRIMARY_MODEL,
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY
        )
        try:
            response = llm.invoke([{
                "role": "user",
                "content": [
                    {"type": "text",      "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                ]
            }])
            return response.content
        except Exception as e:
            return f"Error processing image with Gemini: {str(e)}"


def extract_image_data(image_path: str) -> str:
    """Main function to extract data from an image file path (CLI / workflow usage)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        with open(image_path, "rb") as image_file:
            img_bytes = image_file.read()

        filename = os.path.basename(image_path)
        return _extract_image_text(img_bytes, filename)

    except Exception as e:
        raise Exception(f"Failed to extract data from image: {str(e)}")


def extract_image_data_from_bytes(img_bytes: bytes) -> str:
    """For Streamlit usage when image is already in memory."""
    return _extract_image_text(img_bytes, filename="image.png")
