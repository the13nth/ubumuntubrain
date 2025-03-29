import chromadb
from chromadb.config import Settings
import os
from sentence_transformers import SentenceTransformer

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
client = chromadb.Client(Settings(
    persist_directory="chroma_db",
    anonymized_telemetry=False
))

# Create or get collections for different types of data
text_collection = client.get_or_create_collection(
    name="text_queries",
    metadata={"hnsw:space": "cosine"}
)

image_collection = client.get_or_create_collection(
    name="image_queries",
    metadata={"hnsw:space": "cosine"}
)

document_collection = client.get_or_create_collection(
    name="document_queries",
    metadata={"hnsw:space": "cosine"}
)

location_collection = client.get_or_create_collection(
    name="location_queries",
    metadata={"hnsw:space": "cosine"}
)

def get_text_embedding(text):
    return model.encode(text).tolist()

def get_image_embedding(image_path):
    # For now, we'll use a placeholder for image embeddings
    # In a real application, you'd want to use a proper image embedding model
    return model.encode("image placeholder").tolist()

def store_query(query, query_type, metadata=None):
    if query_type == "text":
        embedding = get_text_embedding(query)
        text_collection.add(
            embeddings=[embedding],
            documents=[query],
            metadatas=[metadata] if metadata else None,
            ids=[f"text_{len(text_collection.get()['ids'])}"]
        )
    elif query_type == "image":
        embedding = get_image_embedding(query)
        image_collection.add(
            embeddings=[embedding],
            documents=[query],
            metadatas=[metadata] if metadata else None,
            ids=[f"image_{len(image_collection.get()['ids'])}"]
        )
    elif query_type == "document":
        embedding = get_text_embedding(query)
        document_collection.add(
            embeddings=[embedding],
            documents=[query],
            metadatas=[metadata] if metadata else None,
            ids=[f"doc_{len(document_collection.get()['ids'])}"]
        )
    elif query_type == "location":
        location_collection.add(
            embeddings=[get_text_embedding(f"location: {query}")],
            documents=[query],
            metadatas=[metadata] if metadata else None,
            ids=[f"loc_{len(location_collection.get()['ids'])}"]
        )

def search_similar(query, query_type, n_results=5):
    if query_type == "text":
        results = text_collection.query(
            query_embeddings=[get_text_embedding(query)],
            n_results=n_results
        )
    elif query_type == "image":
        results = image_collection.query(
            query_embeddings=[get_image_embedding(query)],
            n_results=n_results
        )
    elif query_type == "document":
        results = document_collection.query(
            query_embeddings=[get_text_embedding(query)],
            n_results=n_results
        )
    elif query_type == "location":
        results = location_collection.query(
            query_embeddings=[get_text_embedding(f"location: {query}")],
            n_results=n_results
        )
    else:
        return None
    
    return results 