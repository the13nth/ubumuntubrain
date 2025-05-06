from flask import jsonify
import google.generativeai as genai
from ..core.firebase import get_firebase_db
from ..services.pinecone_service import PineconeService
import numpy as np
from app import model  # Import the SentenceTransformer model instance

class QueryService:
    @staticmethod
    def process_search_query(query, context_type=None):
        try:
            print(f"[QueryService] Received query: {query}")
            pinecone_service = PineconeService()
            
            # Generate query embedding using the SentenceTransformer model
            query_embedding = model.encode(query).tolist()
            print(f"[QueryService] Query embedding (first 5 values): {query_embedding[:5]}")
            
            # Search in Pinecone with the query embedding
            results = pinecone_service.query(query_embedding, top_k=5)
            print(f"[QueryService] Pinecone query results: {results}")
            
            # Process results and generate recommendations
            relevant_docs = []
            if not results or not results.get('ids') or not results['ids'][0]:
                print("[QueryService] No results returned from Pinecone.")
            else:
                for idx, doc_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][idx]
                    text = results['documents'][0][idx]
                    distance = results['distances'][0][idx]
                    print(f"[QueryService] Result idx={idx}, doc_id={doc_id}, type={metadata.get('type')}, distance={distance}")
                    # Filter by context type if specified
                    if context_type and metadata.get('type') != context_type:
                        print(f"[QueryService] Skipping doc_id={doc_id} due to context_type filter: {context_type}")
                        continue
                    # Calculate relevance score (convert distance to similarity)
                    relevance = 1 - (distance / 2)  # Normalize distance to [0,1] range
                    relevant_docs.append({
                        'text': text,
                        'metadata': metadata,
                        'relevance': relevance,
                        'distance': distance
                    })
            print(f"[QueryService] Total relevant docs after filtering: {len(relevant_docs)}")
            # Sort by relevance
            relevant_docs.sort(key=lambda x: x['relevance'], reverse=True)
            # Generate recommendations based on relevant documents
            recommendations = QueryService.generate_context_recommendations(
                query, relevant_docs, context_type
            )
            print(f"[QueryService] Recommendations generated: {len(recommendations)}")
            return {
                'query': query,
                'relevant_documents': relevant_docs,
                'recommendations': recommendations
            }
        except Exception as e:
            print(f"[QueryService] Error in process_search_query: {str(e)}")
            raise Exception(f"Error processing search query: {str(e)}")
    
    @staticmethod
    def generate_context_recommendations(query, context_texts, context_type):
        try:
            # Initialize Gemini model
            model = genai.GenerativeModel('gemini-pro')
            
            # Create RAG prompt with relevance scores
            prompt = QueryService.create_rag_prompt(query, context_texts)
            
            # Generate response
            response = model.generate_content(prompt)
            
            # Process and structure the response
            recommendations = []
            if response.text:
                # Parse the response and extract recommendations
                recommendations = [
                    {
                        'text': rec.strip(),
                        'relevance': 1.0  # Default relevance for generated recommendations
                    } 
                    for rec in response.text.split('\n') 
                    if rec.strip()
                ]
            
            return recommendations
            
        except Exception as e:
            raise Exception(f"Error generating recommendations: {str(e)}")
    
    @staticmethod
    def create_rag_prompt(query, relevant_docs):
        # Create a prompt for RAG-based query processing
        context_texts = []
        for doc in relevant_docs:
            relevance = doc['relevance']
            context_texts.append(
                f"Context (Relevance: {relevance:.2f}): {doc['text']}\n"
                f"Type: {doc['metadata'].get('type', 'unknown')}"
            )
        
        context_str = "\n\n".join(context_texts)
        
        prompt = f"""Based on the following contexts and query, generate relevant recommendations.
        
Query: {query}

Relevant Contexts (sorted by relevance):
{context_str}

Please provide specific, actionable recommendations based on the query and the provided contexts.
Format each recommendation on a new line.
Consider the relevance scores when generating recommendations - prioritize information from more relevant contexts.
"""
        return prompt 