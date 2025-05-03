import os
import google.generativeai as genai
from dotenv import load_dotenv

_model = None

def initialize_gemini():
    """Initialize Gemini model."""
    global _model
    
    try:
        load_dotenv()
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 1024,
            "candidate_count": 1
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        # Try different model versions in case one fails
        try:
            _model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        except Exception as e:
            print(f"Failed to initialize gemini-1.5-flash: {str(e)}")
            try:
                _model = genai.GenerativeModel(
                    model_name="gemini-1.0-pro",
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
            except Exception as e:
                print(f"Failed to initialize gemini-1.0-pro: {str(e)}")
                try:
                    _model = genai.GenerativeModel(
                        model_name="gemini-pro",
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                except Exception as e:
                    print(f"Failed to initialize gemini-pro: {str(e)}")
                    _model = None
        
    except Exception as e:
        print(f"Error initializing Gemini: {str(e)}")
        _model = None

def get_gemini_model():
    """Get the Gemini model instance."""
    global _model
    if _model is None:
        initialize_gemini()
    return _model 