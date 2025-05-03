import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from app.config.settings import Config

class ChromaService:
    """Service for managing ChromaDB operations."""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.model = None
        self.initialize()
    
    def initialize(self):
        """Initialize ChromaDB and sentence transformer model."""
        try:
            print("Initializing ChromaDB...")
            self.client = chromadb.Client(Settings(**Config.CHROMA_SETTINGS))
            print("ChromaDB client created successfully")
            
            # Create or get the collection
            self.collection = self.client.get_or_create_collection(
                name="text_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Collection initialized. Current count: {len(self.collection.get(['documents'])['ids'])}")
            
            # Initialize the sentence transformer model
            self.model = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
            print("Sentence transformer model initialized")
            
        except Exception as e:
            print(f"Error initializing ChromaDB: {str(e)}")
            self.client = None
            self.collection = None
            self.model = None
    
    def add_document(self, text, metadata, doc_id=None):
        """Add a document to ChromaDB."""
        try:
            if not self.collection or not self.model:
                return False
                
            # Generate embedding
            embedding = self.model.encode(text).tolist()
            
            # Generate ID if not provided
            if not doc_id:
                doc_id = f"{metadata.get('type', 'doc')}_{metadata.get('id', 'unknown')}".replace(' ', '_').lower()
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                ids=[doc_id],
                metadatas=[metadata]
            )
            return True
            
        except Exception as e:
            print(f"Error adding document to ChromaDB: {str(e)}")
            return False
    
    def get_all_documents(self, include=['documents', 'metadatas']):
        """Get all documents from ChromaDB."""
        try:
            if not self.collection:
                return None
            return self.collection.get(include=include)
        except Exception as e:
            print(f"Error getting documents from ChromaDB: {str(e)}")
            return None
    
    def search_documents(self, query, n_results=5):
        """Search for similar documents."""
        try:
            if not self.collection or not self.model:
                return None
                
            # Generate query embedding
            query_embedding = self.model.encode(query).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['embeddings', 'documents', 'metadatas', 'distances']
            )
            return results
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return None
    
    def delete_document(self, doc_id):
        """Delete a document from ChromaDB."""
        try:
            if not self.collection:
                return False
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting document: {str(e)}")
            return False 