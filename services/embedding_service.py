from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple


class EmbeddingService:
    """
    Service class for generating and managing embeddings.
    Single Responsibility: Handle all embedding-related operations
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding service with the specified model."""
        self.model = SentenceTransformer(model_name)
        print(f"Embedding model '{model_name}' initialized successfully")
    
    def encode(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for a single text or list of texts."""
        if isinstance(text, str):
            return self.model.encode(text).tolist()
        else:
            return [self.model.encode(t).tolist() for t in text]
    
    def check_existing_embedding(self, collection, metadata: Dict[str, Any]) -> bool:
        """Check if an embedding with the same metadata already exists."""
        try:
            # Get all documents and check metadata manually since ChromaDB where clause is limited
            results = collection.get(include=["metadatas"])
            if not results or 'metadatas' not in results:
                return False
                
            # Check each metadata for matching id and type
            for meta in results['metadatas']:
                if (meta.get('id') == metadata.get('id') and 
                    meta.get('type') == metadata.get('type')):
                    return True
            return False
        except Exception as e:
            print(f"Error checking existing embedding: {str(e)}")
            return False
    
    def add_to_collection(self, collection, text: str, metadata: Dict[str, Any]) -> bool:
        """Add text to collection if it doesn't already exist."""
        try:
            # Check if embedding already exists
            if self.check_existing_embedding(collection, metadata):
                print(f"Embedding already exists for {metadata.get('type')} with ID {metadata.get('id')}")
                return False
                
            # Generate embedding
            embedding = self.encode(text)
            
            # Generate a unique document ID based on metadata
            doc_id = f"{metadata.get('type', 'doc')}_{metadata.get('id', 'unknown')}".replace(' ', '_').lower()
            
            # Add to collection
            collection.add(
                embeddings=[embedding],
                documents=[text],
                ids=[doc_id],
                metadatas=[metadata]
            )
            print(f"Added new embedding for {metadata.get('type')} with ID {metadata.get('id')}")
            return True
        except Exception as e:
            print(f"Error adding to embeddings: {str(e)}")
            return False
    
    def search(self, collection, query: str, n_results: int = 10) -> Dict[str, Any]:
        """Search for similar documents in the collection."""
        try:
            # Generate query embedding
            query_embedding = self.encode(query)
            
            # Search in collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['embeddings', 'documents', 'metadatas', 'distances']
            )
            
            return results
        except Exception as e:
            print(f"Error searching embeddings: {str(e)}")
            return {} 