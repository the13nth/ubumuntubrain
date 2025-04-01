#!/usr/bin/env python
"""
UbumuntuBrain startup script.
This script initializes all services and starts the Flask application.
"""
import os
from initialize_services import get_services
from app import app

def initialize():
    """Initialize all services before starting the app."""
    print("Initializing UbumuntuBrain services...")
    
    # Get the service manager
    services = get_services()
    
    # Print initialization status
    print(f"Embedding Service: {services.embedding_service.__class__.__name__}")
    print(f"Database Service: {services.db_service.__class__.__name__}")
    print(f"Recommendation Service: {services.recommendation_service.__class__.__name__}")
    
    print("Services initialized successfully!")
    return True

if __name__ == "__main__":
    # Initialize services
    if initialize():
        # Get port from environment or use default
        port = int(os.environ.get("PORT", 5000))
        
        # Start the Flask app
        print(f"Starting UbumuntuBrain on port {port}...")
        app.run(debug=True, host="0.0.0.0", port=port) 