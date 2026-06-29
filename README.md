# 🛣️ Highway Cross-Section Analyzer

An AI-powered and Computer Vision-assisted web application designed to analyze highway earthwork cross-section engineering drawings, identify Existing Ground (EG) and Proposed Grade (PG) profiles, classify Cut/Fill regions, and generate color-filled visual overlays.

---

## 🎯 Purpose of the Module

In highway construction, engineers analyze transverse cross-sections to estimate the volume of earthwork (Cut & Fill) required along a road alignment. This module automates the visual inspection and verification process by:
1. **Profile Segmentation**: Distinguishing between the natural terrain (Existing Ground - dashed/dotted lines) and design templates (Proposed Grade - solid lines).
2. **Region Classification**: Segmenting the cross-section into:
   - **FILL**: Proposed Grade lies above the Existing Ground (represented in **Green**).
   - **CUT**: Proposed Grade lies below the Existing Ground (represented in **Red**).
3. **Visual Validation Overlay**: Dynamically rendering a semi-transparent color mask on top of the original CAD or PDF drawing to visually highlight earthwork zones for engineering review.
4. **Local Pixel Quantification**: Utilizing OpenCV-based connected components and grid detection algorithms as an alternative route to measure grid coordinates and calculate local areas.

---

## 👥 Team Contributions

* **Muni Kumar Netlapalli** :
  - Designed and built the interactive web dashboard using **Streamlit**.
  - Developed the document ingestion module utilizing **PyMuPDF (fitz)** to convert scale-accurate engineering PDFs into high-DPI image buffers.
  - Implemented the multimodal AI integration (`agent.py`) using **Google Gemini 2.5 Flash** and **Groq (Qwen/Llama)** models for semantic line style and elevation classification.
  - Authored the visual color-filling drawing engine using **SciPy**'s linear interpolation (`interp1d`) and **PIL**'s alpha-blended image composition.
  - Researched and prototyped the OpenCV-based grid detection and profile isolation scripts (`area_calculator.py`).

---

## 📅 Project Status

### ✅ Completed
- [x] **Streamlit UI Interface**: Functional double-tabbed layout supporting page-by-page PDF rendering and direct PNG/JPG image upload.
- [x] **Multimodal AI Profiling**: Structured API communication with fallback handling between Google Gemini 2.5 Flash and Groq vision models.
- [x] **Scipy coordinate interpolation**: Successful horizontal mapping of sparse coordinates to a full-width pixel grid.
- [x] **PIL Transparency Overlay**: Dynamic rendering of red/green vertical column fills mapping to elevation differences.
- [x] **OpenCV Grid Detection**: Algorithmic grid unit measurement (pixel width of a 10ft × 10ft grid square) and morphological grid line removal.

### 🚧 In Progress
- [ ] **Adaptive Border & Margin Alignment**: Implementing automatic drawing-area cropping to align the LLM's normalized `0-1000` coordinate space with physical drawing boundaries (excluding margins/title blocks).
- [ ] **Hybrid Pipeline Integration**: Merging the high-level semantic labeling of the LLM with local OpenCV contour-tracing to completely eliminate coordinate hallucination and produce pixel-perfect fills.
- [ ] **High-Resolution Detail Preservation**: Improving image preprocessing to ensure thin dashed lines (1-2px) do not merge or blur during API compression.

---

## ⚙️ Dependencies & Setup Instructions

### System Requirements
- Python 3.9+
- Microsoft Windows

### Dependencies
The project relies on the following key Python libraries:
- `streamlit` - Frontend Web App Framework
- `pymupdf` (fitz) - PDF parsing and image rendering
- `scipy` - Scientific utilities (used for linear coordinate interpolation)
- `numpy` - Vector and pixel matrix operations
- `Pillow` (PIL) - Image manipulation and overlay blending
- `opencv-python` - Morphological operations and line masking
- `langchain-google-genai` - LangChain Google AI integration
- `python-dotenv` - Environment variable management
- `langgraph` - Graph execution blueprints

### Installation & Run Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd infra_llm-main
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   ```

5. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Feasibility Report (LLM Coordinate Tracing vs. OpenCV)

### Background
The initial design goal relied on sending the image to a multimodal Vision LLM (e.g., Gemini 2.5 Flash) to return exact coordinates for tracing ground profiles and generating color overlays. During implementation and evaluation, a technical feasibility assessment was conducted.

### Key Findings
1. **Coordinate Hallucination (Low Feasibility)**:
   - *Problem*: Multimodal LLMs do not inspect individual pixel matrices; they construct coordinates based on relative visual predictions. This causes slight coordinate shifts.
   - *Impact*: In engineering drawings where lines are tightly spaced, even a 1.5% coordinate error results in a 30-40 pixel misalignment, making the color overlay spill over boundaries.
2. **Downscaling Loss (Medium Feasibility)**:
   - *Problem*: To fit within API payload limits, large engineering sheets must be resized. Resizing thin dashed lines (1-2px wide) smears or deletes the broken pattern.
   - *Impact*: The model struggles to distinguish dashed (Existing Ground) from solid (Proposed Grade) lines on downscaled sheets.
3. **Semantic Classification (High Feasibility)**:
   - *Problem*: Traditional Computer Vision struggles to understand context (e.g., distinguishing notes from lines, interpreting custom line type symbols).
   - *Impact*: LLMs excel at this task, accurately identifying the number of CUT/FILL zones and finding sheet stations.
