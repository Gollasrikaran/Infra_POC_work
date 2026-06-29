from typing import Dict, TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()

import os
PRIMARY_MODEL  = os.getenv("GEMINI_MODEL")

api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model=PRIMARY_MODEL,
    temperature=0.7,
    google_api_key=api_key
)


class ImageExtractionState(TypedDict):
    image_path: str
    extracted_data: str
    error: str


def supervisor_node(state: ImageExtractionState) -> ImageExtractionState:
    """Supervisor node that routes to appropriate agent."""
    return state


def image_extraction_node(state: ImageExtractionState) -> ImageExtractionState:
    """Image extraction node that processes the image."""
    from agent import extract_image_data
    
    try:
        extracted_data = extract_image_data(state["image_path"])
        state["extracted_data"] = extracted_data
        state["error"] = ""
    except Exception as e:
        state["error"] = str(e)
        state["extracted_data"] = ""
    
    return state


def build_graph():
    """Build the workflow graph for image extraction."""
    workflow = StateGraph(ImageExtractionState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("image_extractor", image_extraction_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add edges
    workflow.add_edge("supervisor", "image_extractor")
    workflow.add_edge("image_extractor", END)
    
    # Compile the graph
    app = workflow.compile()
    return app