import os
from pinecone import Pinecone, ServerlessSpec

class PineconeService:
    def __init__(self):
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = "embed"
        dimension = 384  # all-MiniLM-L6-v2
        region = "us-east-1"  # Update if needed
        cloud = "aws"  # Update if needed
        self.pc = Pinecone(api_key=api_key)
        # Create index if it doesn't exist
        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
        self.index = self.pc.Index(index_name)

    def upsert(self, doc_id, embedding, metadata):
        # Pinecone expects metadata as a dict
        self.index.upsert([(doc_id, embedding, metadata)])

    def query(self, embedding, top_k=5):
        return self.index.query(vector=embedding, top_k=top_k, include_metadata=True)

    def delete(self, doc_id):
        self.index.delete(ids=[doc_id])

    def fetch(self, doc_id):
        """Fetch a single vector by ID."""
        return self.index.fetch(ids=[doc_id])

    def batch_fetch(self, doc_ids, batch_size=10):
        """Fetch multiple vectors in batches to avoid ID length limits."""
        all_results = {'vectors': {}}
        
        # Process IDs in batches
        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i:i + batch_size]
            try:
                batch_results = self.index.fetch(ids=batch_ids)
                # Use attribute access for FetchResponse
                if batch_results and hasattr(batch_results, 'vectors') and batch_results.vectors:
                    all_results['vectors'].update(batch_results.vectors)
            except Exception as e:
                print(f"Error fetching batch {i//batch_size + 1}: {str(e)}")
                continue
                
        return all_results

    def list_vectors(self):
        """Get all vector IDs from the index."""
        try:
            # Get index stats to know how many vectors we have
            stats = self.index.describe_index_stats()
            total_vectors = stats.namespaces.get('', {}).get('vector_count', 0)
            
            if total_vectors == 0:
                return []
            
            # Use query with a zero vector to get all vectors
            # This is a workaround since Pinecone doesn't provide a direct list_vectors method
            zero_vector = [0.0] * 384  # dimension of our embeddings
            results = self.index.query(
                vector=zero_vector,
                top_k=total_vectors,
                include_metadata=True
            )
            
            # Extract IDs from results
            return [match.id for match in results.matches]
            
        except Exception as e:
            print(f"Error listing vectors: {str(e)}")
            return []

# Singleton instance
pinecone_service = PineconeService() 