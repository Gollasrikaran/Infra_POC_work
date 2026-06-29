# Image Data Extraction Tool

A web-based tool powered by Google Gemini AI to extract data from charts, graphs, and visual content using LangGraph workflow.

## Features

- 🖼️ **Image Upload**: Support for PNG, JPG, and JPEG formats
- 🤖 **AI-Powered Extraction**: Uses Google Gemini 2.5 Flash for intelligent data extraction
- 📊 **Comprehensive Analysis**: Extracts chart types, axes, values, trends, and text
- 🌐 **Web Interface**: Beautiful Streamlit UI with responsive design
- ⚡ **Workflow Architecture**: Built with LangGraph for scalable processing

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Make sure your `.env` file contains:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```

## Usage

### Web Interface (Recommended)

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown (usually `http://localhost:8501`)

3. **Upload an image**:
   - Click "Choose an image..." or drag and drop
   - Supported formats: PNG, JPG, JPEG

4. **Extract data**:
   - Click the "Extract Data" button
   - Wait for processing (shows spinner)
   - View results in the extracted data section

### Command Line Interface

For CLI usage, run:
```bash
python main.py
```

Then enter the image file path when prompted.

## Project Structure

```
infra_llm/
├── app.py              # Streamlit web interface
├── main.py             # CLI interface
├── workflow.py         # LangGraph workflow definition
├── agent.py            # Image processing agent
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
└── README.md          # This file
```

## Workflow Architecture

The application uses a LangGraph workflow with the following nodes:

1. **Supervisor Node**: Routes the workflow
2. **Image Extractor Node**: Processes the image using Google Gemini

```
Input → Supervisor → Image Extractor → Output
```

## Dependencies

- `langchain-google-genai`: Google Gemini integration
- `langgraph`: Workflow orchestration
- `streamlit`: Web interface framework
- `python-dotenv`: Environment variable management
- `Pillow`: Image processing

## API Requirements

- **Google Gemini API Key**: Required for image analysis
- Get your API key from [Google AI Studio](https://aistudio.google.com/)

## Supported Image Types

- 📊 Charts and graphs
- 📈 Data visualizations
- 📋 Tables and diagrams
- 📝 Text-heavy images
- 🎯 Statistical plots

## Error Handling

The application includes comprehensive error handling for:
- Invalid file formats
- Missing API keys
- Network issues
- Processing errors

## Development

To extend the application:

1. **Add new extraction types**: Modify the prompt in `agent.py`
2. **Add workflow nodes**: Update `workflow.py` with new nodes and edges
3. **Customize UI**: Modify `app.py` for different Streamlit components

## License

This project is for educational and demonstration purposes.
