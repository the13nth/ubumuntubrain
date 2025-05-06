from flask import Flask
from flask_sock import Sock
from flask_cors import CORS
import google.generativeai as genai
import os

from .config import Config
from .core.database import db
from .core.firebase import firebase

def create_app():
    # Initialize Flask app
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Enable CORS
    CORS(app)
    
    # Initialize WebSocket
    sock = Sock(app)
    
    # Initialize Gemini
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    
    # Try different model versions in case one fails
    model_gemini = None
    for model_name in Config.GEMINI_MODEL_VERSIONS:
        try:
            model_gemini = genai.GenerativeModel(
                model_name=model_name,
                generation_config=Config.GENERATION_CONFIG,
                safety_settings=Config.SAFETY_SETTINGS
            )
            break
        except Exception as e:
            print(f"Failed to initialize {model_name}: {str(e)}")
    
    # Create uploads directory if it doesn't exist
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    # Register blueprints
    from .api.health import health_bp
    from .api.work import work_bp
    from .api.commute import commute_bp
    from .api.search import search_bp
    from .api.tools import tools_bp
    
    app.register_blueprint(health_bp, url_prefix='/api/health')
    app.register_blueprint(work_bp, url_prefix='/api/work')
    app.register_blueprint(commute_bp, url_prefix='/api/commute')
    app.register_blueprint(search_bp, url_prefix='/api/search')
    app.register_blueprint(tools_bp, url_prefix='/api/tools')
    
    # Store WebSocket connections
    app.ws_connections = set()
    
    return app 