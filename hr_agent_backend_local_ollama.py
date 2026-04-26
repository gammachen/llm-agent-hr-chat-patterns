# Simplified HR Agent Backend using Ollama directly
import requests
import json

def ollama_chat(prompt: str, model: str = "qwen2:latest") -> str:
    """Send a chat request to Ollama and return the response."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "Sorry, I couldn't generate a response.")
    except Exception as e:
        return f"Error: {str(e)}"

def get_response(user_input: str) -> str:
    """Get a response from the HR agent."""
    
    # Create a context-aware prompt
    system_prompt = """You are a helpful HR assistant. You can help with:
    - Employee policies and procedures
    - Timekeeping and leave questions
    - General HR inquiries
    
    Be professional, helpful, and concise in your responses."""
    
    full_prompt = f"{system_prompt}\n\nUser: {user_input}\n\nAssistant:"
    
    # Get response from Ollama
    response = ollama_chat(full_prompt)
    
    return response.strip()
