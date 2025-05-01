import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from ..config import Config

class ChromaClient:
    def __init__(self):
        self.client = None
        self.collection = None
        self.model = None
        self.initialize()
    
    def initialize(self):
        """Initialize ChromaDB and sentence transformer model"""
        try:
            print("Initializing ChromaDB...")
            self.client = chromadb.Client(Settings(**Config.CHROMA_SETTINGS))
            print("ChromaDB client created successfully")
            
            # Create or get the collection
            self.collection = self.client.get_or_create_collection(
                name="text_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Collection initialized. Current count: {len(self.collection.get()['ids'])}")
            
            # Initialize the sentence transformer model
            self.model = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
            print("Sentence transformer model initialized")
            
        except Exception as e:
            print(f"Error initializing ChromaDB: {str(e)}")
            raise e
    
    def add_embedding(self, text, metadata, doc_id=None):
        """Add text embedding to ChromaDB"""
        try:
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
            print(f"Error adding embedding: {str(e)}")
            return False
    
    def search(self, query, n_results=5, where=None):
        """Search for similar embeddings"""
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=['embeddings', 'documents', 'metadatas', 'distances']
            )
            
            return results
            
        except Exception as e:
            print(f"Error searching embeddings: {str(e)}")
            return None
    
    def delete_embedding(self, doc_id):
        """Delete embedding from ChromaDB"""
        try:
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            print(f"Error deleting embedding: {str(e)}")
            return False
    
    def get_embedding(self, doc_id):
        """Get specific embedding by ID"""
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas']
            )
            
            if not result or not result['ids']:
                return None
                
            return {
                "id": doc_id,
                "document": result['documents'][0],
                "metadata": result['metadatas'][0]
            }
            
        except Exception as e:
            print(f"Error getting embedding: {str(e)}")
            return None

# Create global ChromaDB instance
chroma = ChromaClient() 