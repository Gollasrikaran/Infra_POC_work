import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import base64

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PRIMARY_MODEL  = os.getenv("GEMINI_MODEL")


PROMPT = """You are an expert highway engineer analyzing a cross-section drawing.

NOTE: This image has been pre-processed with computer vision. Cross-section regions
are color-coded:
  - RED shaded area = CUT region (existing ground above proposed grade — excavation)
  - GREEN shaded area = FILL region (proposed grade above existing ground — embankment)
  - BLUE highlighted line = solid proposed grade line

Use these colors to quickly identify CUT vs FILL regions. Still extract all
coordinates, vertices, and data as instructed below.

GRID SCALE (read from the axis labels on the drawing):
- X-axis (horizontal): Each labeled interval = 10 feet (offset from centerline)
- Y-axis (vertical): Each labeled interval = 10 feet (elevation)
- The X-axis zero is at the centerline; negative values are to the LEFT, positive to the RIGHT.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — IDENTIFY LINE TYPES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- DOTTED/DASHED line = Existing Ground (natural surface). This line spans the full drawing width.
- SOLID line (with slope labels like 2:1, 3:1, 6.0%, 8.0%) = Proposed Grade (design template).
  ⚠️ The SOLID proposed grade line has a LIMITED horizontal extent — it does NOT span the full drawing.
  It starts at the LEFT catch point and ends at the RIGHT catch point where it meets the dashed line.
  DO NOT assume the proposed grade continues beyond where the solid line visually ends.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — FIND ALL CATCH POINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scan the entire solid line carefully from LEFT to RIGHT.
A catch point = any location where the SOLID proposed grade line physically meets OR crosses
the DASHED existing ground line.

There may be 2, 3, or even 4 catch points — creating multiple enclosed regions.
Record EVERY catch point in order from left to right with its (x, elevation).

  ✅ LEFT catch point  = leftmost point where solid line touches/crosses dashed line
  ✅ RIGHT catch point = rightmost point where solid line touches/crosses dashed line
  ✅ MIDDLE catch points = any intermediate crossings between left and right

  ⚠️ CRITICAL RULES:
  - If the solid line ends before reaching the right edge of the drawing, the right catch point
    is where the solid line ENDS (meets the dashed line), NOT the drawing edge.
  - DO NOT extend the proposed grade line beyond where it visually ends in the drawing.

  ⚠️ VISUAL BOUNDARY WARNING (For both Colored and Black-and-White drawings):
  - A region terminates immediately wherever the solid proposed line meets or crosses the dashed ground line.
  - If the drawing has color shading, any transition between GREEN (Fill) and RED (Cut) is a catch point.
  - If the drawing has NO color (empty space), a catch point occurs every single time the solid line and dashed line physically intersect or kiss. Do NOT bypass an intersection point to group separate areas together.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — IDENTIFY EACH ENCLOSED REGION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Between every pair of consecutive catch points there is one enclosed region.
For each region, check which line is higher INSIDE that region:
  • DOTTED above SOLID inside the region → CUT  (excavation needed)
  • SOLID above DOTTED inside the region → FILL (embankment needed)

Do NOT rely on x=0 alone — a region entirely on the left side (e.g. x=-50 to x=-10)
can still be CUT or FILL based on which line is higher within that specific zone.

Also note which side of the centerline (x=0) the region falls on:
  - Both catch points x < 0 → side = "LEFT"
  - Both catch points x > 0 → side = "RIGHT"  
  - One negative, one positive → side = "CROSSES_CENTERLINE"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — EXTRACT POLYGON VERTICES FOR EACH REGION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For EACH enclosed region, list the ordered (x, elevation) coordinates
that trace its perimeter. Go CLOCKWISE:
  1. Start at the LEFT catch point.
  2. Follow the PROPOSED GRADE line rightward to the RIGHT catch point,
     recording every labelled or visually readable vertex along the way.
  3. Follow the EXISTING GROUND line leftward back to the LEFT catch point,
     recording every readable vertex along that path.
  4. The polygon is now closed — do NOT repeat the start point.

  ⚠️ STRICT RULES:
  - Include as many intermediate vertices as the drawing allows — more points = more accurate area.
  - Only include x-offsets that fall STRICTLY within that region's own two catch points.
  - Do NOT calculate the area yourself — leave total_area_sqft as null.
    The area will be computed by a precise Python math function on the server.

Also extract globally:
  - Station number (e.g. 13+00, 20+50)
  - Key elevations (centerline existing, centerline proposed, each catch point elevation)
  - All slope ratios visible on the solid proposed grade line (e.g. 2:1, 4:1, 8.0%)
  - The full x-extent of the solid proposed grade line

    - NEVER cross through, over, or under the flat finished roadway surface template to connect a left-side area directly to a right-side area inside a single polygon. 
  - The flat road surface serves as a strict structural divider. 
  - Every time an enclosed shape or shaded region closes up, the polygon MUST terminate immediately at that specific coordinate. 
  - For example, if a left slope hits the road shoulder near x = -20 ft, close the loop there. It must NOT inherit any coordinates past that structural boundary.

    ⚠️ CRITICAL: DISTINGUISH GRID LINES FROM GROUND LINES
  - The background contains solid or faint horizontal and vertical grid lines (like the 710 elevation line). NEVER use these straight grid lines as a boundary for a polygon.
  - The existing ground line is exclusively the WAVY, UNDULATING dashed/dotted line. It is almost never a perfectly flat, straight horizontal line.
  - If you trace a path and find that the elevation remains exactly the same number (e.g., 710) across multiple horizontal steps, STOP. You are accidentally tracking a background grid line instead of the true ground profile.



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT — Respond ONLY with valid JSON, no markdown, no code fences, no extra text:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{
  "station": "station number if visible, else null",
  "slopes": "all slope ratios visible on the solid proposed grade line",
  "proposed_grade_extent": "e.g. solid line from x=-50 ft to x=-5 ft",
  "existing_ground_line": "brief description of dashed line behaviour across the drawing",
  "total_intersections": <integer — number of enclosed regions found>,
  "intersections": [
    {
      "id": 1,
      "type": "CUT or FILL",
      "side": "LEFT or RIGHT or CROSSES_CENTERLINE",
      "left_catch_point":  {"x": <ft>, "elevation": <ft>},
      "right_catch_point": {"x": <ft>, "elevation": <ft>},
      "area_calculation": {
        "method": "shoelace",
        "vertices": [
          {"x": <ft>, "elevation": <ft>}
        ],
        "total_area_sqft": null
      },
      "notes": "brief observation for this specific region"
    }
  ],
  "reasoning": "walk through every catch point found left to right, explain why each region is CUT or FILL, and list the polygon vertices you extracted for each region",
  "notes": "any other overall observations"
}"""


STATION_ONLY_PROMPT = """You are an expert highway engineer. Scan the drawing to find and extract ONLY the station number (e.g. 12+50, 13+00, 105+25).
Respond with ONLY a JSON object, no markdown code fences:
{
  "station": "station number if visible, else null"
}"""


def _extract_image_text(img_bytes: bytes, filename: str) -> str:
    """Send image to Gemini and return raw JSON string."""
    # Pass the AQ token string straight to the google_api_key field.
    # The SDK core handles bearer string conversion natively without Pydantic proxy crashes.
    llm = ChatGoogleGenerativeAI(
        model=PRIMARY_MODEL,
        temperature=0.2,          
        google_api_key=GOOGLE_API_KEY
    )

    image_base64 = base64.b64encode(img_bytes).decode("utf-8")

    try:
        response = llm.invoke([{
            "role": "user",
            "content": [
                {"type": "text",      "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }])
        content = response.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block["text"]
            return str(content)  
        return content  
    except Exception as e:
        return f"Error processing image: {str(e)}"


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


def extract_station_only_from_bytes(img_bytes: bytes) -> str:
    """Send image to Gemini and return ONLY the station number in JSON."""
    llm = ChatGoogleGenerativeAI(
        model=PRIMARY_MODEL,
        temperature=0.1,          
        google_api_key=GOOGLE_API_KEY
    )

    image_base64 = base64.b64encode(img_bytes).decode("utf-8")

    try:
        response = llm.invoke([{
            "role": "user",
            "content": [
                {"type": "text",      "text": STATION_ONLY_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        }])
        content = response.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block["text"]
            return str(content)
        return content
    except Exception as e:
        return f"Error processing image: {str(e)}"
