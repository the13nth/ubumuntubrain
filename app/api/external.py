from flask import Blueprint, request, jsonify
from functools import wraps
import os

from ..services.pinecone_service import pinecone_service
from ..utils.websocket import ws_manager

# Create blueprint
external_bp = Blueprint('external', __name__)

def require_api_key(f):
    """Decorator to require API key for external endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.getenv('EXTERNAL_API_KEY'):
            return jsonify({"error": "Invalid API key"}), 401
        return f(*args, **kwargs)
    return decorated_function

@external_bp.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Query is required'}), 400
        
    try:
        search_results = pinecone_service.query(data['query'])
        return jsonify(search_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@external_bp.route('/query', methods=['POST'])
@require_api_key
def external_query():
    """External API endpoint for queries."""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        # Process the query using the search service
        try:
            # Search for relevant documents
            search_results = pinecone_service.query(data['query'])
            
            # Generate response using Gemini
            context = None
            if search_results and search_results['documents']:
                context = "\n".join(search_results['documents'][0])
            
            response = gemini_service.generate_response(data['query'], context)
            
            result = {
                "answer": response,
                "relevant_docs": search_results['documents'] if search_results else []
            }
            
            # Notify WebSocket clients about the API call
            ws_manager.broadcast({
                "type": "api_call",
                "query": data['query'],
                "query_result": result
            })
            
            return jsonify(result)
            
        except Exception as e:
            error_message = str(e)
            print(f"Error processing query: {error_message}")
            
            # Send error notification to WebSocket clients
            ws_manager.broadcast({
                "type": "api_call_error",
                "query": data['query'],
                "error": error_message
            })
            
            return jsonify({
                "error": error_message,
                "answer": f"Error processing query: {error_message}"
            }), 500
    
    except Exception as e:
        error_message = str(e)
        print(f"External API error: {error_message}")
        return jsonify({"error": error_message}), 500 