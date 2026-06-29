import os
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def list_models():
    if not GROQ_API_KEY:
        print("GROQ_API_KEY is not set.")
        return
        
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get("https://api.groq.com/openai/v1/models", headers=headers)
        response.raise_for_status()
        models = response.json().get("data", [])
        print("Available Groq Models:")
        for m in sorted(models, key=lambda x: x["id"]):
            print(f"- {m['id']} (owned by: {m.get('owned_by')})")
    except Exception as e:
        print(f"Error: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"Response: {response.text}")

if __name__ == "__main__":
    list_models()
