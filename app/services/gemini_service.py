import google.generativeai as genai
from ..config.settings import Config

class GeminiService:
    """Service for managing Gemini API operations."""
    
    def __init__(self):
        self.model = None
        self.initialize()
    
    def initialize(self):
        """Initialize Gemini API client."""
        try:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
            print("Gemini API initialized successfully")
        except Exception as e:
            print(f"Error initializing Gemini API: {str(e)}")
            self.model = None
    
    def generate_response(self, prompt, context=None):
        """Generate a response using Gemini."""
        try:
            if not self.model:
                return None
                
            # Prepare the chat
            chat = self.model.start_chat(
                history=[],
                generation_config=Config.GEMINI_CONFIG,
                safety_settings=Config.SAFETY_SETTINGS
            )
            
            # Add context if provided
            if context:
                chat.send_message(f"Context: {context}")
            
            # Generate response
            response = chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            print(f"Error generating Gemini response: {str(e)}")
            return None
    
    def analyze_text(self, text, instruction):
        """Analyze text with specific instructions."""
        try:
            if not self.model:
                return None
                
            prompt = f"""
            Text to analyze: {text}
            
            Instructions: {instruction}
            
            Please provide your analysis:
            """
            
            response = self.model.generate_content(
                prompt,
                generation_config=Config.GEMINI_CONFIG,
                safety_settings=Config.SAFETY_SETTINGS
            )
            
            return response.text
            
        except Exception as e:
            print(f"Error analyzing text with Gemini: {str(e)}")
            return None 