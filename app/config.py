import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Flask Config
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev')
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json', 'xlsx', 'csv'}
    
    # Firebase Config
    FIREBASE_KEY_PATH = 'firebase-key.json'
    
    # Gemini Config
    GENERATION_CONFIG = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 1024,
        "candidate_count": 1
    }
    
    SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    # Model Config
    SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
    GEMINI_MODEL_VERSIONS = ['gemini-1.5-flash', 'gemini-1.0-pro', 'gemini-pro'] 