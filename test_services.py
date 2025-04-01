"""
Test script to verify the services are working correctly.
Run this script to check if all services are properly initialized and can perform their basic functions.
"""
import json
import os
from initialize_services import get_services

def test_embedding_service():
    """Test the embedding service."""
    print("\n--- Testing Embedding Service ---")
    services = get_services()
    embedding_service = services.embedding_service
    
    # Test encoding a single text
    text = "This is a test sentence for embedding."
    embedding = embedding_service.encode(text)
    print(f"Single text embedding length: {len(embedding)}")
    
    # Test encoding multiple texts
    texts = ["This is the first test sentence.", "This is the second test sentence."]
    embeddings = embedding_service.encode(texts)
    print(f"Multiple text embedding count: {len(embeddings)}")
    print(f"First embedding length: {len(embeddings[0])}")
    
    return True

def test_chromadb_service():
    """Test the ChromaDB service."""
    print("\n--- Testing ChromaDB Service ---")
    services = get_services()
    db_service = services.db_service
    
    # Test initialization
    if not db_service.initialized:
        success = db_service.initialize()
        print(f"DB initialization: {'Success' if success else 'Failed'}")
    else:
        print("DB already initialized")
    
    # Test getting collection
    collection = db_service.get_collection()
    print(f"Collection items: {len(collection.get()['ids'])}")
    
    # Test getting all embeddings
    all_data = db_service.get_all_embeddings()
    if all_data and 'ids' in all_data:
        print(f"Retrieved {len(all_data['ids'])} documents from ChromaDB")
    else:
        print("No embeddings found or error retrieving them")
    
    return True

def test_recommendation_service():
    """Test the recommendation service."""
    print("\n--- Testing Recommendation Service ---")
    services = get_services()
    recommendation_service = services.recommendation_service
    
    # Test adding a document
    test_doc = "This is a test document for the recommendation service."
    test_metadata = {"source": "Test", "type": "general"}
    success = recommendation_service.add_document(test_doc, test_metadata)
    print(f"Add document: {'Success' if success else 'Failed'}")
    
    # Test getting recommendations
    test_query = "test document"
    recommendations = recommendation_service.get_recommendations(test_query)
    print(f"Recommendations count: {recommendations.count}")
    
    # Print first recommendation if available
    if recommendations.count > 0:
        first_rec = recommendations.recommendations[0]
        print(f"First recommendation: {first_rec.text[:50]}... (score: {first_rec.score:.4f})")
        print(f"Metadata: {first_rec.metadata.source}, {first_rec.metadata.type}")
    
    # Test saving recommendations
    if recommendations.count > 0:
        test_file = recommendation_service.save_health_recommendations(test_query)
        if test_file:
            print(f"Recommendations saved to: {test_file}")
            
            # Verify the file exists and has content
            if os.path.exists(test_file):
                with open(test_file, 'r') as f:
                    data = json.load(f)
                    print(f"Saved {data.get('count', 0)} recommendations")
            else:
                print(f"File not found: {test_file}")
        else:
            print("Failed to save recommendations")
    
    return True

def run_all_tests():
    """Run all tests."""
    print("=== Starting Service Tests ===")
    
    try:
        # Test all services
        embedding_test = test_embedding_service()
        db_test = test_chromadb_service()
        recommendation_test = test_recommendation_service()
        
        # Print summary
        print("\n=== Test Summary ===")
        print(f"Embedding Service: {'PASSED' if embedding_test else 'FAILED'}")
        print(f"ChromaDB Service: {'PASSED' if db_test else 'FAILED'}")
        print(f"Recommendation Service: {'PASSED' if recommendation_test else 'FAILED'}")
        
        if embedding_test and db_test and recommendation_test:
            print("\n✅ All tests passed!")
            return True
        else:
            print("\n❌ Some tests failed!")
            return False
    except Exception as e:
        print(f"\n❌ Error during tests: {str(e)}")
        return False

if __name__ == "__main__":
    run_all_tests() 