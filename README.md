# UbumuntuBrain

A Flask application that provides AI-powered recommendations and search using vector embeddings.

## Project Structure

This project follows the SOLID principles for better maintainability and separation of concerns:

- **Single Responsibility**: Each class has one responsibility and one reason to change
- **Open/Closed**: Classes are open for extension but closed for modification
- **Liskov Substitution**: Services use interface patterns for substitution
- **Interface Segregation**: Services depend on specific interfaces rather than large ones
- **Dependency Inversion**: High-level modules depend on abstractions, not concrete implementations

### Directory Structure

```
├── app.py                    # Main Flask application entry point
├── initialize_services.py    # Service initialization manager
├── services/                 # Service implementations
│   ├── __init__.py
│   ├── pinecone_service.py   # Pinecone database service
│   └── recommendation_service.py  # Recommendation management
├── models/                   # Data models
│   ├── __init__.py
│   └── recommendation.py     # Recommendation data models
├── utils/                    # Utility functions
│   ├── __init__.py
│   └── recommendation_utils.py  # Helper functions
└── static/                   # Static assets for web interface
```

## Getting Started

### Prerequisites

- Python 3.8+
- Pinecone
- Sentence Transformers
- Google Gemini API
- Firebase (optional)

### Installation

1. Clone the repository
2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:

   ```
   GOOGLE_API_KEY=your_google_api_key
   EXTERNAL_API_KEY=your_api_key
   ```

### Running the Application

Start the application:

```
python app.py
```

The application will be available at <http://localhost:5000>

## Services

### EmbeddingService

Handles the generation and management of vector embeddings using Sentence Transformers.

```python
# Example usage
from services.embedding_service import EmbeddingService

embedding_service = EmbeddingService()
embedding = embedding_service.encode("Your text here")
```

### PineconeService

Manages the vector database operations.

```python
# Example usage
from services.pinecone_service import PineconeService

db_service = PineconeService()
```

### RecommendationService

Combines embedding and database services to provide recommendations.

```python
# Example usage
from services.recommendation_service import RecommendationService

recommendation_service = RecommendationService(embedding_service, db_service)
results = recommendation_service.get_recommendations("Your query")
```

## API Endpoints

- `POST /api/search`: Search for recommendations
- `POST /api/create`: Add a new embedding
- `POST /api/upload`: Upload and process a file
- `GET /api/data`: Get all stored data

## License

MIT
