import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Union, Tuple


class ChromaDBService:
    """
    Service class for managing ChromaDB operations.
    Single Responsibility: Handle all ChromaDB-related operations
    """
    
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize the ChromaDB service with the specified persistence directory."""
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize the ChromaDB client and collection."""
        try:
            print("Initializing ChromaDB...")
            self.client = chromadb.Client(Settings(
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            ))
            print("ChromaDB client created successfully")

            # Create or get the collection
            self.collection = self.client.get_or_create_collection(
                name="text_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Collection initialized. Current count: {len(self.collection.get()['ids'])}")
            
            self.initialized = True
            return True
        except Exception as e:
            print(f"Error initializing ChromaDB: {str(e)}")
            self.initialized = False
            return False
    
    def get_collection(self):
        """Get the current collection, initializing if necessary."""
        if not self.initialized:
            self.initialize()
        return self.collection
    
    def add_test_data(self, embedding_service) -> bool:
        """Add test data to the collection if it's empty."""
        try:
            if not self.initialized:
                self.initialize()
                
            if len(self.collection.get()['ids']) == 0:
                print("Adding test data to empty collection...")
                test_data = [
                    "This is a test document about health context.",
                    "This is a test document about work context.",
                    "This is a test document about commute context."
                ]
                
                embeddings = [embedding_service.encode(text) for text in test_data]
                self.collection.add(
                    embeddings=embeddings,
                    documents=test_data,
                    ids=[f"test_{i+1}" for i in range(len(test_data))],
                    metadatas=[
                        {"source": "Test", "type": "health"},
                        {"source": "Test", "type": "work"},
                        {"source": "Test", "type": "commute"}
                    ]
                )
                print(f"Added {len(test_data)} test documents")
                return True
            return False
        except Exception as e:
            print(f"Error adding test data: {str(e)}")
            return False
    
    def add_document(self, embedding: List[float], document: str, doc_id: str, metadata: Dict[str, Any]) -> bool:
        """Add a document to the collection."""
        try:
            if not self.initialized:
                self.initialize()
                
            self.collection.add(
                embeddings=[embedding],
                documents=[document],
                ids=[doc_id],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            print(f"Error adding document to ChromaDB: {str(e)}")
            return False
    
    def search(self, query_embedding: List[float], n_results: int = 10) -> Dict[str, Any]:
        """Search the collection using the query embedding."""
        try:
            if not self.initialized:
                self.initialize()
                
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['embeddings', 'documents', 'metadatas', 'distances']
            )
            return results
        except Exception as e:
            print(f"Error searching ChromaDB: {str(e)}")
            return {}
    
    def get_all_embeddings(self):
        """Get all embeddings from the collection."""
        try:
            if not self.initialized:
                self.initialize()
                
            return self.collection.get(include=['embeddings', 'documents', 'metadatas'])
        except Exception as e:
            print(f"Error getting all embeddings: {str(e)}")
            return {} 