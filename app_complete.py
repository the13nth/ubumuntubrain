from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.decomposition import PCA
import PyPDF2
import io
import google.generativeai as genai
from dotenv import load_dotenv
from flask_sock import Sock
import json
import threading
from functools import wraps
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import sqlite3

# Import our services manager
from initialize_services import get_services

# Load environment variables
load_dotenv()

# Initialize Firebase
try:
    cred = credentials.Certificate('firebase-key.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Error initializing Firebase: {str(e)}")
    db = None

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)
sock = Sock(app)

# Store WebSocket connections
ws_connections = set()
ws_lock = threading.Lock()

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Get services from the service manager
services = get_services()
embedding_service = services.embedding_service
db_service = services.db_service
recommendation_service = services.recommendation_service

# Initialize Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 1024,
    "candidate_count": 1
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Try different model versions in case one fails
try:
    model_gemini = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)
except Exception as e:
    print(f"Failed to initialize gemini-1.5-flash: {str(e)}")
    try:
        model_gemini = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                         generation_config=generation_config,
                                         safety_settings=safety_settings)
    except Exception as e:
        print(f"Failed to initialize gemini-1.0-pro: {str(e)}")
        try:
            model_gemini = genai.GenerativeModel(model_name="gemini-pro",
                                             generation_config=generation_config,
                                             safety_settings=safety_settings)
        except Exception as e:
            print(f"Failed to initialize gemini-pro: {str(e)}")
            model_gemini = None  # Will use fallback responses

def broadcast_to_websockets(message):
    """Broadcast a message to all connected WebSocket clients."""
    with ws_lock:
        dead_sockets = set()
        for ws in ws_connections:
            try:
                ws.send(json.dumps(message))
            except Exception:
                dead_sockets.add(ws)
        
        # Remove dead connections
        for dead_ws in dead_sockets:
            ws_connections.remove(dead_ws)

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.getenv('EXTERNAL_API_KEY'):
            return jsonify({"error": "Invalid API key"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/external/query', methods=['POST'])
@require_api_key
def external_query():
    """External API endpoint for queries."""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        # Process the query using the existing search function
        try:
            result = process_search_query(data['query'])
            
            # Notify WebSocket clients about the API call
            broadcast_to_websockets({
                "type": "api_call",
                "query": data['query'],
                "query_result": result
            })
            
            return jsonify(result)
            
        except Exception as e:
            error_message = str(e)
            print(f"Error processing query: {error_message}")
            
            # Send error notification to WebSocket clients
            broadcast_to_websockets({
                "type": "api_call_error",
                "query": data['query'],
                "error": error_message
            })
            
            return jsonify({
                "error": error_message,
                "answer": f"Error processing query: {error_message}",
                "query_embedding_visualization": {"x": 0, "y": 0, "z": 0}
            }), 500
    
    except Exception as e:
        error_message = str(e)
        print(f"External API error: {error_message}")
        return jsonify({"error": error_message}), 500

def allowed_file(filename):
    """Check if an uploaded file is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_json(file):
    """Extract text from a JSON file."""
    data = json.loads(file.read().decode('utf-8'))
    # If it's a simple string, return it
    if isinstance(data, str):
        return data
    # If it's a dictionary, try to extract 'text' field, or concatenate all string values
    elif isinstance(data, dict):
        if 'text' in data:
            return data['text']
        else:
            return ' '.join([str(value) for value in data.values() if isinstance(value, str)])
    # If it's a list, try to join all string elements
    elif isinstance(data, list):
        return ' '.join([str(item) for item in data if isinstance(item, str)])
    return json.dumps(data)  # Fallback to JSON string

def process_embeddings_for_visualization(all_embeddings, query_embedding, results):
    """Process embeddings to create a visualization."""
    try:
        # Combine all embeddings including the query
        combined_embeddings = all_embeddings + [query_embedding]
        
        # Use PCA to reduce dimensions to 3 for visualization
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(combined_embeddings)
        
        # Extract query embedding visualization (last item)
        query_viz = {
            "x": float(embeddings_3d[-1][0]),
            "y": float(embeddings_3d[-1][1]),
            "z": float(embeddings_3d[-1][2])
        }
        
        # Process document embeddings
        doc_viz = []
        
        # Get the number of results returned and process only those that match
        if results and 'documents' in results and results['documents']:
            result_count = len(results['documents'][0])
            
            # Get indices of results in the all_embeddings list
            result_indices = []
            for i in range(result_count):
                # This is a simplification; in a real-world app, you'd need 
                # a more robust way to match results with original embeddings
                # Here we assume the order matches
                result_indices.append(i % len(all_embeddings))
            
            # Create visualization data for matching documents
            for i, idx in enumerate(result_indices):
                if idx < len(embeddings_3d) - 1:  # Exclude the query
                    doc_data = {
                        "x": float(embeddings_3d[idx][0]),
                        "y": float(embeddings_3d[idx][1]),
                        "z": float(embeddings_3d[idx][2]),
                        "text": results['documents'][0][i][:50] + "..." if len(results['documents'][0][i]) > 50 else results['documents'][0][i],
                        "context_type": results['metadatas'][0][i].get('type', 'unknown') if i < len(results['metadatas'][0]) else 'unknown'
                    }
                    doc_viz.append(doc_data)
        
        return {
            "query": query_viz,
            "documents": doc_viz
        }
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        return {
            "query": {"x": 0, "y": 0, "z": 0},
            "documents": []
        }

def process_search_query(query):
    """Process a search query and return the results."""
    # Generate query embedding and search
    query_embedding = embedding_service.encode(query)
    
    # Search in ChromaDB
    results = db_service.search(query_embedding, 10)  # Increased to get more relevant contexts
    
    # Make sure we have results
    if not results or 'documents' not in results or not results['documents'] or not results['documents'][0]:
        return {
            "answer": "No documents found in the database.",
            "query_embedding_visualization": {"x": 0, "y": 0, "z": 0}
        }
    
    # Get all embeddings including the query for PCA
    all_data = db_service.get_all_embeddings()
    if not all_data or 'embeddings' not in all_data or not all_data['embeddings']:
        return {
            "answer": "No documents found in the database.",
            "query_embedding_visualization": {"x": 0, "y": 0, "z": 0}
        }
    
    # Process embeddings and get visualization data
    visualization_data = process_embeddings_for_visualization(
        all_data['embeddings'],
        query_embedding,
        results
    )
    
    # Process matching contexts with relevance scores
    matching_contexts = []
    for i in range(len(results['documents'][0])):
        # Convert distance to similarity score (1 - normalized_distance)
        similarity = 1 - min(results['distances'][0][i], 1.0)  # Ensure distance is not > 1
        context_type = results['metadatas'][0][i].get('type', 'unknown')
        
        matching_contexts.append({
            'type': context_type,
            'text': results['documents'][0][i],
            'relevance': similarity,
            'metadata': results['metadatas'][0][i]
        })
    
    # Sort contexts by relevance
    matching_contexts.sort(key=lambda x: x['relevance'], reverse=True)
    
    # Extract the top context texts
    context_texts = [c['text'] for c in matching_contexts[:5]]
    context_types = [c['type'] for c in matching_contexts[:5]]
    
    # Generate answer based on contexts
    answer = generate_answer(query, context_texts, context_types)
    
    # Return the results
    return {
        "answer": answer,
        "matching_contexts": matching_contexts,
        "query_embedding_visualization": visualization_data
    }

def generate_answer(query, context_texts, context_types):
    """Generate an answer based on the query and relevant contexts."""
    if not context_texts:
        return "I couldn't find any relevant information to answer your question."
    
    # Create a prompt for the AI model
    prompt = f"""
    Based on the following information, please provide a comprehensive answer to the question: "{query}"
    
    Relevant information:
    {" ".join([f"- {text}" for text in context_texts])}
    
    Instructions:
    1. Answer the question accurately and clearly.
    2. Use only information from the provided context.
    3. If the context doesn't contain enough information to answer the question, say so.
    4. Include relevant quotes from the context to support your answer.
    5. If the question is about specific context types: {", ".join(set(context_types))}, focus on that information.
    """
    
    try:
        if model_gemini:
            response = model_gemini.generate_content(prompt)
            return response.text if response.text else "I couldn't generate a response based on the available information."
        else:
            return "AI model is not available. Please try again later."
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        return f"I couldn't generate a response due to an error: {str(e)}"

@app.route('/api/search', methods=['POST'])
def search():
    """Search API endpoint."""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        query = data['query']
        
        # Process the search query
        result = process_search_query(query)
        
        # Save recommendations for different contexts
        try:
            health_file = recommendation_service.save_health_recommendations(query)
            work_file = recommendation_service.save_work_recommendations(query)
            commute_file = recommendation_service.save_commute_recommendations(query)
            
            if health_file or work_file or commute_file:
                result["recommendations_saved"] = {
                    "health": health_file,
                    "work": work_file,
                    "commute": commute_file
                }
        except Exception as e:
            print(f"Error saving recommendations: {str(e)}")
        
        return jsonify(result)
    except Exception as e:
        error_message = str(e)
        print(f"Search error: {error_message}")
        return jsonify({"error": error_message}), 500

@app.route('/api/create', methods=['POST'])
def create_embedding():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        text = data.get('text')
        if not text or not text.strip():
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        # Create metadata
        metadata = {"source": "Manual Input"}
        
        # Add optional metadata fields if provided
        if 'type' in data:
            metadata['type'] = data['type']
        if 'source' in data:
            metadata['source'] = data['source']
        
        # Add document using our service
        success = recommendation_service.add_document(text.strip(), metadata)
        
        return jsonify({"success": success})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Process the file based on its type
            if file.filename.endswith('.pdf'):
                text = extract_text_from_pdf(file)
            elif file.filename.endswith('.json'):
                text = extract_text_from_json(file)
            else:  # .txt file
                text = file.read().decode('utf-8')
            
            # Create metadata
            metadata = {"source": f"File: {file.filename}"}
            
            # Add document using our service
            success = recommendation_service.add_document(text, metadata)
            
            return jsonify({"success": success})
            
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    
    return jsonify({"success": False, "error": "Invalid file type"}), 400

@app.route('/api/data')
def get_data():
    try:
        print("Fetching data from ChromaDB...")
        # Get all data from ChromaDB with embeddings included
        data = db_service.get_all_embeddings()
        print(f"Retrieved {len(data['ids'] if 'ids' in data else [])} documents from ChromaDB")
        
        # If there's no data, return an empty list
        if not data or 'ids' not in data:
            print("No data found in ChromaDB")
            return jsonify([])
        
        # Format the data for the table
        formatted_data = []
        for i in range(len(data['ids'])):
            formatted_data.append({
                'id': data['ids'][i],
                'document': data['documents'][i][:100] + '...' if len(data['documents'][i]) > 100 else data['documents'][i],
                'embedding_size': len(data['embeddings'][i]),
                'source': data.get('metadatas', [{}])[i].get('source', 'Manual Input') if data.get('metadatas') else 'Manual Input'
            })
        
        print(f"Formatted {len(formatted_data)} documents for display")
        return jsonify(formatted_data)
    except Exception as e:
        print(f"Error in get_data: {str(e)}")
        return jsonify([]), 500

# Connect to WebSocket
@sock.route('/ws')
def websocket(ws):
    with ws_lock:
        ws_connections.add(ws)
        print(f"New WebSocket connection. Total connections: {len(ws_connections)}")
    
    try:
        # Keep connection open and handle messages
        while True:
            message = ws.receive()
            if message:
                try:
                    msg_data = json.loads(message)
                    # Handle different message types
                    if msg_data.get('type') == 'ping':
                        ws.send(json.dumps({"type": "pong"}))
                except Exception as e:
                    print(f"Error processing WebSocket message: {str(e)}")
    except Exception as e:
        print(f"WebSocket disconnected: {str(e)}")
    finally:
        with ws_lock:
            if ws in ws_connections:
                ws_connections.remove(ws)
                print(f"WebSocket disconnected. Remaining connections: {len(ws_connections)}")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000) 