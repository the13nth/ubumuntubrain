"""
Service initialization and configuration module.
This module provides a central location for initializing all services.
"""
import os
from services.embedding_service import EmbeddingService
from services.chromadb_service import ChromaDBService
from services.recommendation_service import RecommendationService
from models.recommendation import Recommendation, Metadata, RecommendationSet

class ServiceManager:
    """
    Manages the initialization and access to various services in the application.
    Follows the Singleton pattern to ensure services are only initialized once.
    """
    _instance = None
    
    def __new__(cls):
        """Create a singleton instance of ServiceManager."""
        if cls._instance is None:
            cls._instance = super(ServiceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the service manager if it hasn't been initialized yet."""
        if not self._initialized:
            # Create data directories
            os.makedirs("data/recommendations", exist_ok=True)
            os.makedirs("uploads", exist_ok=True)
            
            # Initialize services
            self._embedding_service = EmbeddingService('all-MiniLM-L6-v2')
            self._db_service = ChromaDBService("chroma_db")
            self._db_service.initialize()
            
            # Initialize recommendation service with dependencies
            self._recommendation_service = RecommendationService(
                self._embedding_service, 
                self._db_service
            )
            
            # Add test data if needed
            self._db_service.add_test_data(self._embedding_service)
            
            self._initialized = True
            print("All services initialized successfully")
    
    @property
    def embedding_service(self) -> EmbeddingService:
        """Get the embedding service instance."""
        return self._embedding_service
    
    @property
    def db_service(self) -> ChromaDBService:
        """Get the database service instance."""
        return self._db_service
    
    @property
    def recommendation_service(self) -> RecommendationService:
        """Get the recommendation service instance."""
        return self._recommendation_service
    
    def initialize_all(self) -> bool:
        """Initialize or reinitialize all services."""
        try:
            # Reinitialize the database
            success = self._db_service.initialize()
            if not success:
                print("Failed to initialize database service")
                return False
            
            # Add test data if needed
            self._db_service.add_test_data(self._embedding_service)
            
            print("All services reinitialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing services: {str(e)}")
            return False

# Singleton instance for importing elsewhere
service_manager = ServiceManager()

def get_services():
    """Get the service manager instance."""
    return service_manager

if __name__ == "__main__":
    # When run directly, initialize all services and print status
    manager = get_services()
    print(f"Embedding Service: {manager.embedding_service.__class__.__name__}")
    print(f"DB Service: {manager.db_service.__class__.__name__}")
    print(f"Recommendation Service: {manager.recommendation_service.__class__.__name__}")
    print("Services initialization complete") 