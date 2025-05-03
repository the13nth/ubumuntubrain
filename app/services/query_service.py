from flask import jsonify
import google.generativeai as genai
from ..core.chroma import get_chroma_collection
from ..core.firebase import get_firebase_db

class QueryService:
    @staticmethod
    def process_search_query(query, context_type=None):
        try:
            collection = get_chroma_collection()
            
            # Search in ChromaDB
            results = collection.query(
                query_texts=[query],
                n_results=5
            )
            
            # Process results and generate recommendations
            relevant_docs = []
            for idx, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][idx]
                text = results['documents'][0][idx]
                
                # Filter by context type if specified
                if context_type and metadata.get('type') != context_type:
                    continue
                    
                relevant_docs.append({
                    'text': text,
                    'metadata': metadata,
                    'distance': results['distances'][0][idx]
                })
            
            # Generate recommendations based on relevant documents
            recommendations = QueryService.generate_context_recommendations(
                query, relevant_docs, context_type
            )
            
            return {
                'query': query,
                'relevant_documents': relevant_docs,
                'recommendations': recommendations
            }
            
        except Exception as e:
            raise Exception(f"Error processing search query: {str(e)}")
    
    @staticmethod
    def generate_context_recommendations(query, context_texts, context_type):
        try:
            # Initialize Gemini model
            model = genai.GenerativeModel('gemini-pro')
            
            # Create RAG prompt
            prompt = QueryService.create_rag_prompt(query, context_texts)
            
            # Generate response
            response = model.generate_content(prompt)
            
            # Process and structure the response
            recommendations = []
            if response.text:
                # Parse the response and extract recommendations
                # This is a simplified version - actual implementation would need
                # more robust parsing based on the expected response format
                recommendations = [
                    {'text': rec.strip()} 
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
            context_texts.append(f"Context: {doc['text']}\nType: {doc['metadata'].get('type', 'unknown')}")
        
        context_str = "\n\n".join(context_texts)
        
        prompt = f"""Based on the following contexts and query, generate relevant recommendations.
        
Query: {query}

Relevant Contexts:
{context_str}

Please provide specific, actionable recommendations based on the query and the provided contexts.
Format each recommendation on a new line.
"""
        return prompt 