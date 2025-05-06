import warnings
# Filter out the specific FutureWarning from transformers
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers.utils.generic')

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
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
import datetime
import sqlite3
import traceback
import umap
from app.services.visualization_service import VisualizationService
from app.services.pinecone_service import pinecone_service
from app.config.settings import Config
from app.services.firebase_service import firebase_service
import logging
from firebase_admin import firestore
from app.services.document_service import DocumentService


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')
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

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize services
firebase_service.initialize()
db = firebase_service.db
storage = firebase_service.storage

# Initialize the sentence transformer model
model = SentenceTransformer(Config.SENTENCE_TRANSFORMER_MODEL)
logger.info("Sentence transformer model initialized")

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
    logger.error(f"Failed to initialize gemini-1.5-flash: {str(e)}")
    try:
        model_gemini = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                         generation_config=generation_config,
                                         safety_settings=safety_settings)
    except Exception as e:
        logger.error(f"Failed to initialize gemini-1.0-pro: {str(e)}")
        try:
            model_gemini = genai.GenerativeModel(model_name="gemini-pro",
                                             generation_config=generation_config,
                                             safety_settings=safety_settings)
        except Exception as e:
            logger.error(f"Failed to initialize gemini-pro: {str(e)}")
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
            logger.error(f"Error processing query: {error_message}")
            
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
        logger.error(f"External API error: {error_message}")
        return jsonify({"error": error_message}), 500

@sock.route('/ws')
def handle_websocket(ws):
    """Handle WebSocket connections."""
    with ws_lock:
        ws_connections.add(ws)
    try:
        while True:
            # Keep the connection alive
            ws.receive()
    except Exception:
        with ws_lock:
            ws_connections.remove(ws)

def process_search_query(query, context_type=None):
    print(f"[app.py] Received query: {query}")
    # Generate query embedding
    query_embedding = model.encode(query).tolist()
    print(f"[app.py] Query embedding (first 5 values): {query_embedding[:5]}")
    # Search in Pinecone with context type filter if provided
    if context_type:
        results = pinecone_service.query(query_embedding, top_k=5)
    else:
        results = pinecone_service.query(query_embedding, top_k=5)
    print(f"[app.py] Pinecone query results: {results}")
    matches = results.get('matches', [])
    if not matches:
        print(f"[app.py] No results found for query: {query}")
        return {
            "answer": f"No {context_type if context_type else ''} documents found in the database.",
            "recommendations": []
        }
    relevant_docs = []
    for idx, match in enumerate(matches):
        meta = match.get('metadata', {})
        doc_type = meta.get('type', '')
        doc_id = meta.get('id', '')
        text_snippet = ''
        # Fetch full document content from Firebase if type is 'document'
        if doc_type == 'document' and doc_id:
            try:
                text_snippet = DocumentService.get_document(doc_id)
            except Exception as e:
                print(f"[app.py] Error fetching document content for {doc_id}: {str(e)}")
                text_snippet = meta.get('text', '')
        else:
            text_snippet = meta.get('text', '')
        print(f"[app.py] Result idx={idx}, type={doc_type}, similarity={match.get('score')}, text_snippet={text_snippet[:100]}")
        relevant_docs.append({
            'text': text_snippet,
            'metadata': meta,
            'relevance': match.get('score', 0),
            'distance': 1 - match.get('score', 0)
        })
    print(f"[app.py] Total matching contexts: {len(relevant_docs)}")
    relevant_docs.sort(key=lambda x: x['relevance'], reverse=True)
    # Use only the text for context_texts
    context_texts = [doc['text'] for doc in relevant_docs]
    answer, recommendations = generate_context_recommendations(query, context_texts, context_type)
    print(f"[app.py] Recommendations generated: {len(recommendations)}")
    return {
        'answer': answer,
        'recommendations': recommendations
    }

def generate_context_recommendations(query, context_texts, context_type):
    """Generate context-specific recommendations using Gemini."""
    
    # Get previous recommendations from Firebase
    previous_recommendations = []
    if db is not None:
        try:
            if context_type == 'health':
                prev_recs = db.collection('health_ai_recommendation').order_by('created_at', direction=firestore.Query.DESCENDING).limit(1).get()
            elif context_type == 'work':
                prev_recs = db.collection('work_ai_recommendation').order_by('created_at', direction=firestore.Query.DESCENDING).limit(1).get()
            elif context_type == 'commute':
                prev_recs = db.collection('commute_ai_recommendation').order_by('created_at', direction=firestore.Query.DESCENDING).limit(1).get()
            
            for doc in prev_recs:
                rec_data = doc.to_dict()
                if 'recommendations' in rec_data:
                    previous_recommendations = rec_data['recommendations']
                break
        except Exception as e:
            logger.error(f"Error fetching previous recommendations: {str(e)}")

    # Analyze query and context
    analysis_prompt = f"""You are UbumuntuBrain, an intelligent AI assistant. Analyze this query and context:

Query: {query}

Available Contexts:
{chr(10).join([f"Context {i+1}:{chr(10)}{text}" for i, text in enumerate(context_texts)])}

Previous Recommendations:
{chr(10).join([f"- {rec}" for rec in previous_recommendations]) if previous_recommendations else "No previous recommendations available."}

Analyze the following:
1. What is the user trying to achieve with this query?
2. Which contexts are most relevant to this query?
3. What key information from the contexts helps answer this query?
4. What specific actions or information should be provided?

Provide your analysis in a clear, structured way that will help generate a response."""

    try:
        # Get query analysis
        analysis = model_gemini.generate_content(analysis_prompt)
        if not analysis.text:
            return "I apologize, but I couldn't analyze your query. Would you like to try rephrasing it?", []

        # Build response prompt using the analysis
        response_prompt = f"""You are UbumuntuBrain, a helpful AI assistant. Generate a response based on this context:

Query: {query}

Analysis of the query and context:
{analysis.text}

Previous Recommendations:
{chr(10).join([f"- {rec}" for rec in previous_recommendations]) if previous_recommendations else "No previous recommendations available."}

Current Contexts:
{chr(10).join([f"Context {i+1}:{chr(10)}{text}" for i, text in enumerate(context_texts)])}

Provide a friendly, helpful response that:
1. Acknowledges the user's query
2. References relevant information from the contexts
3. Incorporates previous recommendations if relevant
4. Provides clear, actionable recommendations or next steps
5. Maintains a conversational, supportive tone

Your response should be practical and directly address the user's needs."""

        # Generate the final response
        response = model_gemini.generate_content(response_prompt)
        if not response.text:
            return "I apologize, but I couldn't generate a response. Would you like to try asking in a different way?", []

        # Extract recommendations from the response
        text = response.text
        recommendations = []
        
        # Look for recommendations in the response
        if "recommendations:" in text.lower():
            rec_section = text.lower().split("recommendations:", 1)
            if len(rec_section) > 1:
                rec_lines = rec_section[1].split("\n")
                recommendations = [r.strip("- ").strip() for r in rec_lines if r.strip().startswith("-")]
        elif "next steps:" in text.lower():
            rec_section = text.lower().split("next steps:", 1)
            if len(rec_section) > 1:
                rec_lines = rec_section[1].split("\n")
                recommendations = [r.strip("- ").strip() for r in rec_lines if r.strip().startswith("-")]
        elif "suggested actions:" in text.lower():
            rec_section = text.lower().split("suggested actions:", 1)
            if len(rec_section) > 1:
                rec_lines = rec_section[1].split("\n")
                recommendations = [r.strip("- ").strip() for r in rec_lines if r.strip().startswith("-")]

        return text, recommendations

    except Exception as e:
        logger.error(f"Error in generate_context_recommendations: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"I apologize, but I encountered an error while processing your query. Would you like to try asking in a different way?", []

@app.route('/api/process-contexts', methods=['POST'])
def process_contexts():
    """Process all available contexts and generate recommendations."""
    try:
        results = {
            "health": {"contexts": [], "recommendations": []},
            "work": {"contexts": [], "recommendations": []},
            "commute": {"contexts": [], "recommendations": []}
        }
        
        # Process health contexts
        health_contexts = db.collection('health_context').order_by('created_at', direction=firestore.Query.DESCENDING).limit(3).get()
        for doc in health_contexts:
            context = doc.to_dict()
            query = f"Health status with blood sugar {context.get('bloodSugar')}, exercise {context.get('exerciseMinutes')} minutes"
            result = process_search_query(query, "health")
            results["health"]["contexts"].append({
                "id": doc.id,
                "data": context,
                "analysis": result["answer"],
                "recommendations": result["recommendations"]
            })
        
        # Process work contexts
        work_contexts = db.collection('work_context').order_by('created_at', direction=firestore.Query.DESCENDING).limit(3).get()
        for doc in work_contexts:
            context = doc.to_dict()
            query = f"Work task {context.get('taskName')} with priority {context.get('priority')} and status {context.get('status')}"
            result = process_search_query(query, "work")
            results["work"]["contexts"].append({
                "id": doc.id,
                "data": context,
                "analysis": result["answer"],
                "recommendations": result["recommendations"]
            })
        
        # Process commute contexts
        commute_contexts = db.collection('commute_context').order_by('created_at', direction=firestore.Query.DESCENDING).limit(3).get()
        for doc in commute_contexts:
            context = doc.to_dict()
            query = f"Commute from {context.get('startLocation')} to {context.get('endLocation')} with {context.get('transportMode')}"
            result = process_search_query(query, "commute")
            results["commute"]["contexts"].append({
                "id": doc.id,
                "data": context,
                "analysis": result["answer"],
                "recommendations": result["recommendations"]
            })
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error processing contexts: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Update the existing search endpoint to use the new processing functions
@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get('query')
        context_type = data.get('context_type')  # Optional context type filter
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        result = process_search_query(query, context_type)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return jsonify({"error": str(e)}), 500

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    try:
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(file)
        
        # Extract text from all pages
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def extract_text_from_json(file):
    """Extract text from a JSON file."""
    try:
        # Load and parse JSON
        data = json.loads(file.read().decode('utf-8'))
        
        # Convert JSON to string, handling nested structures
        if isinstance(data, dict):
            # If it's a dictionary, extract all values
            text = "\n".join(str(v) for v in data.values() if v is not None)
        elif isinstance(data, list):
            # If it's a list, extract all items
            text = "\n".join(str(item) for item in data if item is not None)
        else:
            # If it's a simple value, convert to string
            text = str(data)
        
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from JSON: {str(e)}")

def allowed_file(filename):
    """Check if an uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('query.html')

@app.route('/query')
def query_page():
    return render_template('query.html')

@app.route('/create')
def create_page():
    return render_template('create.html')

@app.route('/api/create', methods=['POST'])
def create_embedding():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        text = data.get('text')
        if not text or not text.strip():
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        # Generate embedding
        embedding = model.encode(text.strip()).tolist()
        doc_id = f"text_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
        metadata = {"source": "Manual Input"}
        pinecone_service.upsert(doc_id, embedding, metadata)
        
        return jsonify({"success": True})
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
            # Save file to Firebase Storage
            doc_id = f"doc_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            bucket = storage.bucket()
            blob = bucket.blob(f'uploads/{doc_id}/{secure_filename(file.filename)}')
            file.seek(0)
            blob.upload_from_file(file)
            download_url = blob.generate_signed_url(datetime.timedelta(hours=1))  # or use blob.public_url
            
            # Create metadata with timestamp
            metadata = {
                "source": f"File: {file.filename}",
                "type": "document",
                "created_at": datetime.datetime.utcnow().isoformat(),
                "file_type": file.filename.split('.')[-1],
                "color": "#ea4335",  # Red for documents
                "size_visual": 10
            }
            
            # Store metadata and storage info in Firestore
            if db:
                db.collection('uploaded_documents').document(doc_id).set({
                    'filename': file.filename,
                    'storage_path': blob.name,
                    'download_url': download_url,
                    'metadata': metadata,
                    'created_at': firestore.SERVER_TIMESTAMP
                })
            
            return jsonify({
                "success": True,
                "doc_id": doc_id,
                "message": "File uploaded and stored in Firebase Storage successfully",
                "metadata": metadata,
                "download_url": download_url
            })
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    return jsonify({"success": False, "error": "Invalid file type"}), 400

@app.route('/api/data')
def get_data():
    try:
        logger.info("Fetching data from all sources...")
        
        # Pinecone does not support get all, so this endpoint should be refactored to use tracked IDs or another method.
        # For now, return an empty list or implement your own ID tracking.
        formatted_data = []
        logger.info(f"Formatted {len(formatted_data)} total documents for display")
        return jsonify(formatted_data)
    except Exception as e:
        logger.error(f"Error in get_data: {str(e)}")
        traceback.print_exc()
        return jsonify([])  # Return empty array instead of error

@app.route('/api/embeddings-visualization')
def get_embeddings_visualization():
    print('[Visualization] Fetching data from Pinecone for visualization...')
    response = VisualizationService.get_embeddings_visualization()
    print('[Visualization] Visualization data prepared and returned.')
    return response

def get_point_color(doc_type):
    """Get color for different document types with improved visibility"""
    color_map = {
        'health_context': '#00CC00',          # Brighter green
        'work_context': '#0066FF',            # Brighter blue
        'commute_context': '#FF6600',         # Brighter orange
        'health_recommendation': '#66FF66',    # Lighter bright green
        'work_recommendation': '#66B2FF',      # Lighter bright blue
        'commute_recommendation': '#FFB366',   # Lighter bright orange
        'query': '#FF0000',                   # Bright red
        'response': '#9933FF',                # Bright purple
        'unknown': '#999999'                  # Lighter gray
    }
    return color_map.get(doc_type, color_map['unknown'])

def get_point_size(doc_type):
    """Get size for different point types"""
    size_map = {
        'health_context': 8,
        'work_context': 8,
        'commute_context': 8,
        'health_recommendation': 10,
        'work_recommendation': 10,
        'commute_recommendation': 10,
        'query': 12,
        'response': 12,
        'unknown': 6
    }
    return size_map.get(doc_type, size_map['unknown'])

def create_rag_prompt(query, relevant_docs):
    # For standard RAG queries
    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(relevant_docs)])
    
    # Create the prompt
    prompt = f"""You are a friendly and helpful AI personal assistant named UbumuntuBrain. Your responses should be:
1. Conversational and empathetic
2. Clear and well-structured
3. Based on the provided context and previous recommendations
4. Include specific, actionable suggestions

When analyzing the context, consider:
- Current situation and user needs
- Previous recommendations and their status
- Related health, work, or commute contexts
- Potential connections between different contexts

Structure your response as follows:
1. A brief, friendly acknowledgment of the query
2. Analysis of the current situation
3. Specific recommendations or suggestions
4. Follow-up considerations or next steps

Context:
{context}

Question: {query}

Please provide a helpful, conversational response that makes the user feel understood and supported. If the context doesn't contain enough information, acknowledge that while still being helpful."""

    return prompt

@app.route('/firebase-query')
def fetch_firebase_query():
    """Fetch the latest query from Firebase and display UI to submit it to RAG"""
    try:
        if db is None:
            return render_template('firebase_error.html', error="Firebase not initialized")
        
        # Get the latest query from Firebase
        queries_ref = db.collection('rag_queries')
        query = queries_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).get()
        
        latest_query = None
        for doc in query:
            latest_query = doc.to_dict()
            latest_query['id'] = doc.id
            break
        
        return render_template('firebase_query.html', query=latest_query)
    
    except Exception as e:
        logger.error(f"Error fetching from Firebase: {str(e)}")
        return render_template('firebase_error.html', error=str(e))

@app.route('/submit-firebase-query', methods=['POST'])
def submit_firebase_query():
    """Submit a query from Firebase to the RAG system"""
    try:
        query_text = request.form.get('query')
        if not query_text:
            return jsonify({"error": "No query provided"}), 400
        
        # Process the query
        result = process_search_query(query_text)
        
        # Update the Firebase document with the result
        if db is not None:
            query_id = request.form.get('query_id')
            if query_id:
                db.collection('rag_queries').document(query_id).update({
                    'answer': result['answer'],
                    'processed': True,
                    'processed_timestamp': firestore.SERVER_TIMESTAMP
                })
        
        return render_template('firebase_result.html', result=result, query=query_text)
    
    except Exception as e:
        logger.error(f"Error processing Firebase query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/latest-firebase-query', methods=['GET'])
def latest_firebase_query():
    try:
        # Try local storage first
        local_data = get_from_local_db('query')
        if local_data:
            return jsonify({'query': local_data})
            
        # Check if Firebase is initialized
        if not db:
            raise Exception("Firebase database not initialized")
            
        # Query Firebase
        query_ref = db.collection('rag_queries').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
        docs = query_ref.get()
        
        if not docs:
            return jsonify({'query': None})
            
        query_data = docs[0].to_dict()
        query_data['id'] = docs[0].id
        
        # Save to local storage for quick access
        save_to_local_db('query', query_data)
        
        return jsonify({'query': query_data})
    
    except Exception as e:
        logger.error(f"Error fetching latest query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/latest-health-context', methods=['GET'])
def latest_health_context():
    try:
        if not db:
            raise Exception("Firebase database not initialized")
            
        context_ref = db.collection('health_contexts').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
        docs = context_ref.get()
        
        if not docs:
            return jsonify({'context': None})
            
        context_data = docs[0].to_dict()
        context_data['id'] = docs[0].id
        
        return jsonify({'context': context_data})
            
    except Exception as e:
        logger.error(f"Error fetching health context: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/latest-work-context', methods=['GET'])
def latest_work_context():
    try:
        if not db:
            raise Exception("Firebase database not initialized")
            
        context_ref = db.collection('work_contexts').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
        docs = context_ref.get()
        
        if not docs:
            return jsonify({'context': None})
            
        context_data = docs[0].to_dict()
        context_data['id'] = docs[0].id
        
        return jsonify({'context': context_data})
            
    except Exception as e:
        logger.error(f"Error fetching work context: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/latest-commute-context', methods=['GET'])
def latest_commute_context():
    try:
        if not db:
            raise Exception("Firebase database not initialized")
            
        context_ref = db.collection('commute_contexts').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1)
        docs = context_ref.get()
        
        if not docs:
            return jsonify({'context': None})
            
        context_data = docs[0].to_dict()
        context_data['id'] = docs[0].id
        
        return jsonify({'context': context_data})
            
    except Exception as e:
        logger.error(f"Error fetching commute context: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_layout_response(doc_ref, dashboard_context, raw_response, parsed_json):
    """Helper function to generate formatted layout response and update Firebase"""
    try:
        from datetime import datetime
        import json
        
        # Validate the parsed JSON
        validated = {
            "cols": min(max(1, parsed_json.get('cols', 2)), 12),
            "rows": min(max(1, parsed_json.get('rows', 2)), 20),
            "rowHeight": min(max(40, parsed_json.get('rowHeight', 200)), 200),
            "margin": parsed_json.get('margin', [10, 10])
        }
        
        # Create the final config by merging with original config
        original_config = dashboard_context.get('originalConfig', {})
        final_config = {**original_config}
        for key in validated:
            final_config[key] = validated[key]
        
        # Calculate changes
        changes = {
            "colsChanged": original_config.get('cols') != final_config.get('cols'),
            "rowsChanged": original_config.get('rows') != final_config.get('rows'),
            "rowHeightChanged": original_config.get('rowHeight') != final_config.get('rowHeight'),
            "marginChanged": original_config.get('margin') != final_config.get('margin')
        }
        
        # Format the complete response structure
        formatted_response = {
            "response": {
                "timestamp": datetime.now().isoformat(),
                "raw": raw_response,
                "parsed": parsed_json,
                "validated": validated
            },
            "result": {
                "finalConfig": final_config,
                "changes": changes
            }
        }
        
        # Update Firestore with the formatted response
        doc_ref.update({
            'answer': formatted_response,
            'processed': True,
            'processed_timestamp': firestore.SERVER_TIMESTAMP
        })
        
        # Return the formatted response
        return jsonify(formatted_response)
    except Exception as e:
        logger.error(f"Error in generate_layout_response: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/process-firebase-query', methods=['POST'])
def process_firebase_query_api():
    """API endpoint to process a query from Firebase"""
    try:
        query_text = request.form.get('query')
        if not query_text:
            return jsonify({"error": "No query provided"}), 400
        
        # Process the query using standard RAG
        result = process_search_query(query_text)
        
        # Save the result to Firebase if available
        query_id = request.form.get('query_id')
        if db is not None and query_id:
            try:
                # Save the query result
                doc_ref = db.collection('rag_queries').document(query_id)
                doc_ref.set({
                    'query': query_text,
                    'answer': result['answer'],
                    'processed': True,
                    'processed_timestamp': firestore.SERVER_TIMESTAMP,
                    'matching_contexts': result.get('matching_contexts', [])
                }, merge=True)
                
                # Save individual context responses
                health_context_id = request.form.get('health_context_id')
                work_context_id = request.form.get('work_context_id')
                commute_context_id = request.form.get('commute_context_id')
                
                # Filter matching contexts by type
                health_contexts = [ctx for ctx in result.get('matching_contexts', []) if ctx['type'] == 'health']
                work_contexts = [ctx for ctx in result.get('matching_contexts', []) if ctx['type'] == 'work']
                commute_contexts = [ctx for ctx in result.get('matching_contexts', []) if ctx['type'] == 'commute']
                
                # Save health context response
                if health_context_id and health_contexts:
                    db.collection('health_responses').add({
                        'query_id': query_id,
                        'health_context_id': health_context_id,
                        'contexts': health_contexts,
                        'created_at': firestore.SERVER_TIMESTAMP
                    })
                
                # Save work context response
                if work_context_id and work_contexts:
                    db.collection('work_responses').add({
                        'query_id': query_id,
                        'work_context_id': work_context_id,
                        'contexts': work_contexts,
                        'created_at': firestore.SERVER_TIMESTAMP
                    })
                
                # Save commute context response
                if commute_context_id and commute_contexts:
                    db.collection('commute_responses').add({
                        'query_id': query_id,
                        'commute_context_id': commute_context_id,
                        'contexts': commute_contexts,
                        'created_at': firestore.SERVER_TIMESTAMP
                    })
                
                # Add query_id to the result
                result['query_id'] = query_id
                
            except Exception as e:
                logger.error(f"Error saving to Firebase: {str(e)}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

def save_response_to_firebase(query, raw_response, response_type, original_query_id=None):
    """Save raw RAG response to Firebase for future reference and analysis"""
    if db is None:
        logger.error("Firebase not initialized, cannot save response")
        return
    
    try:
        # Create a new document in the rag_responses collection
        rag_responses_ref = db.collection('rag_responses')
        
        # Prepare the data to be saved
        response_data = {
            'query': query,
            'raw_response': raw_response,
            'response_type': response_type,
            'original_query_id': original_query_id,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'embeddings_count': len(pinecone_service.list_vectors()) if pinecone_service else 0
        }
        
        # Add the document to the collection
        rag_responses_ref.add(response_data)
        logger.info(f"Successfully saved response to rag_responses collection")
    
    except Exception as e:
        logger.error(f"Error saving response to Firebase: {str(e)}")

@app.route('/api/save-recommendations', methods=['POST'])
def save_recommendations():
    """Save AI-generated recommendations to Firebase"""
    if db is None:
        return jsonify({"error": "Firebase not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Create a new document in the health_ai_recommendation collection
        recommendations_ref = db.collection('health_ai_recommendation')
        
        # Prepare the data to be saved
        recommendation_data = {
            'recommendations': data.get('recommendations', []),
            'bloodSugar': data.get('bloodSugar'),
            'exerciseMinutes': data.get('exerciseMinutes'),
            'mealType': data.get('mealType'),
            'medication': data.get('medication'),
            'status': data.get('status'),
            'query_id': data.get('query_id'),
            'health_context_id': data.get('health_context_id'),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': data.get('created_at') or firestore.SERVER_TIMESTAMP
        }
        
        # Add the document to the collection
        doc_ref = recommendations_ref.add(recommendation_data)
        logger.info(f"Successfully saved recommendations to health_ai_recommendation collection with ID: {doc_ref[1].id}")
        
        return jsonify({
            "success": True,
            "message": "Recommendations saved successfully",
            "recommendation_id": doc_ref[1].id
        })
    except Exception as e:
        logger.error(f"Error saving recommendations: {str(e)}")
        return jsonify({"error": f"Error saving recommendations: {str(e)}"}), 500

def sync_local_recommendations_with_firebase():
    """Sync local work recommendations with Firebase"""
    try:
        # Get local recommendations from SQLite
        conn = sqlite3.connect('local_data.db')
        c = conn.cursor()
        c.execute('SELECT * FROM work_context ORDER BY created_at DESC')
        local_recommendations = c.fetchall()
        conn.close()

        if not local_recommendations:
            logger.info("No local recommendations to sync")
            return

        # Get Firebase recommendations
        recommendations_ref = db.collection('work_ai_recommendation')
        
        for local_rec in local_recommendations:
            # Convert SQLite row to dict
            columns = ['id', 'task_name', 'status', 'priority', 'collaborators', 
                      'deadline', 'notes', 'timestamp', 'type', 'created_at']
            rec_dict = dict(zip(columns, local_rec))
            
            # Check if recommendation exists in Firebase
            query = recommendations_ref.where('work_context_id', '==', rec_dict['id']).limit(1).get()
            
            recommendation_data = {
                'taskName': rec_dict['task_name'],
                'status': rec_dict['status'],
                'priority': rec_dict['priority'],
                'collaborators': rec_dict['collaborators'],
                'deadline': rec_dict['deadline'],
                'notes': rec_dict['notes'],
                'work_context_id': rec_dict['id'],
                'timestamp': firestore.SERVER_TIMESTAMP,
                'created_at': firestore.SERVER_TIMESTAMP
            }
            
            if not query:  # Document doesn't exist in Firebase
                logger.info(f"Adding local recommendation {rec_dict['id']} to Firebase")
                doc_ref = recommendations_ref.add(recommendation_data)
                logger.info(f"Created new recommendation in Firebase with ID: {doc_ref[1].id}")
            else:
                # Update existing document
                for doc in query:
                    logger.info(f"Updating existing recommendation in Firebase with ID: {doc.id}")
                    print(f"Updating existing recommendation in Firebase with ID: {doc.id}")
                    doc.reference.update(recommendation_data)
                    
    except Exception as e:
        print(f"Error syncing recommendations with Firebase: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")

@app.route('/api/save-work-recommendations', methods=['POST'])
def save_work_recommendations():
    """Save AI-generated work recommendations to Firebase and embed in RAG"""
    print("Received request to save work recommendations")
    
    if db is None:
        print("Error: Firebase DB is not initialized")
        return jsonify({"error": "Firebase not initialized"}), 500
    
    try:
        data = request.get_json()
        print(f"Received work recommendation data: {json.dumps(data, indent=2)}")
        
        if not data:
            print("Error: No data provided in request")
            return jsonify({"error": "No data provided"}), 400
            
        # Create a new document in the work_ai_recommendation collection
        recommendations_ref = db.collection('work_ai_recommendation')
        
        # Prepare the data to be saved
        recommendation_data = {
            'recommendations': data.get('recommendations', []),
            'taskName': data.get('taskName'),
            'status': data.get('status'),
            'priority': data.get('priority'),
            'collaborators': data.get('collaborators', ''),
            'deadline': data.get('deadline', ''),
            'notes': data.get('notes', ''),
            'query_id': data.get('query_id'),
            'work_context_id': data.get('work_context_id'),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': firestore.SERVER_TIMESTAMP
        }
        
        # First, try to update existing document if work_context_id exists
        doc_id = None
        if data.get('work_context_id'):
            # Query for existing recommendation with same work_context_id
            existing_docs = recommendations_ref.where('work_context_id', '==', data['work_context_id']).limit(1).get()
            
            for doc in existing_docs:
                # Update existing document
                doc.reference.update(recommendation_data)
                doc_id = doc.id
                print(f"Updated existing recommendation document with ID: {doc_id}")
                break
        
        # If no existing document found or no work_context_id, create new document
        if not doc_id:
            doc_ref = recommendations_ref.add(recommendation_data)
            doc_id = doc_ref[1].id
            print(f"Created new recommendation document with ID: {doc_id}")
        
        # Embed in RAG
        text = f"Work Recommendation: {', '.join(data.get('recommendations', []))}"
        metadata = {
            'id': doc_id,
            'type': 'work_recommendation',
            'source': 'AI Recommendation',
            'created_at': firestore.SERVER_TIMESTAMP
        }
        embed_context_in_rag(text, metadata)
        
        return jsonify({
            "success": True,
            "message": "Work recommendations saved successfully",
            "recommendation_id": doc_id
        })
    except Exception as e:
        error_msg = f"Error saving work recommendations: {str(e)}"
        print(f"Error details: {error_msg}")
        print(f"Stack trace: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

@app.route('/api/save-commute-recommendations', methods=['POST'])
def save_commute_recommendations():
    """Save AI-generated commute recommendations to Firebase and embed in RAG"""
    if db is None:
        return jsonify({"error": "Firebase not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Create a new document in the commute_ai_recommendation collection
        recommendations_ref = db.collection('commute_ai_recommendation')
        
        # Prepare the data to be saved
        recommendation_data = {
            'recommendations': data.get('recommendations', []),
            'startLocation': data.get('startLocation'),
            'endLocation': data.get('endLocation'),
            'duration': data.get('duration'),
            'trafficCondition': data.get('trafficCondition'),
            'transportMode': data.get('transportMode'),
            'notes': data.get('notes', ''),
            'query_id': data.get('query_id'),
            'commute_context_id': data.get('commute_context_id'),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': data.get('created_at') or firestore.SERVER_TIMESTAMP
        }
        
        # Add the document to the collection
        doc_ref = recommendations_ref.add(recommendation_data)
        doc_id = doc_ref[1].id
        print(f"Successfully saved commute recommendations to commute_ai_recommendation collection with ID: {doc_id}")
        
        # Embed in RAG
        text = f"Commute Recommendation: {', '.join(data.get('recommendations', []))}"
        metadata = {
            'id': doc_id,
            'type': 'commute_recommendation',
            'source': 'AI Recommendation',
            'created_at': firestore.SERVER_TIMESTAMP
        }
        embed_context_in_rag(text, metadata)
        
        return jsonify({
            "success": True,
            "message": "Commute recommendations saved successfully",
            "recommendation_id": doc_id
        })
    except Exception as e:
        print(f"Error saving commute recommendations: {str(e)}")
        return jsonify({"error": f"Error saving commute recommendations: {str(e)}"}), 500

@app.route('/api/save-health-recommendations', methods=['POST'])
def save_health_recommendations():
    """Save AI-generated health recommendations to Firebase and embed in RAG"""
    if db is None:
        return jsonify({"error": "Firebase not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Create a new document in the health_ai_recommendation collection
        recommendations_ref = db.collection('health_ai_recommendation')
        
        # Prepare the data to be saved
        recommendation_data = {
            'recommendations': data.get('recommendations', []),
            'bloodSugar': data.get('bloodSugar'),
            'exerciseMinutes': data.get('exerciseMinutes'),
            'mealType': data.get('mealType'),
            'medication': data.get('medication'),
            'status': data.get('status'),
            'query_id': data.get('query_id'),
            'health_context_id': data.get('health_context_id'),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': data.get('created_at') or firestore.SERVER_TIMESTAMP
        }
        
        # Add the document to the collection
        doc_ref = recommendations_ref.add(recommendation_data)
        doc_id = doc_ref[1].id
        print(f"Successfully saved recommendations to health_ai_recommendation collection with ID: {doc_id}")
        
        # Embed in RAG
        text = f"Health Recommendation: {', '.join(data.get('recommendations', []))}"
        metadata = {
            'id': doc_id,
            'type': 'health_recommendation',
            'source': 'AI Recommendation',
            'created_at': firestore.SERVER_TIMESTAMP
        }
        embed_context_in_rag(text, metadata)
        
        return jsonify({
            "success": True,
            "message": "Recommendations saved successfully",
            "recommendation_id": doc_id
        })
    except Exception as e:
        print(f"Error saving recommendations: {str(e)}")
        return jsonify({"error": f"Error saving recommendations: {str(e)}"}), 500

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('local_data.db')
    c = conn.cursor()
    
    # Create tables for different types of data
    c.execute('''CREATE TABLE IF NOT EXISTS queries
                 (id TEXT PRIMARY KEY, query TEXT, timestamp TEXT, 
                  source TEXT, processed BOOLEAN, answer TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS health_context
                 (id TEXT PRIMARY KEY, blood_sugar TEXT, created_at TEXT,
                  exercise_minutes TEXT, meal_type TEXT, medication TEXT,
                  notes TEXT, timestamp TEXT, type TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS work_context
                 (id TEXT PRIMARY KEY, task_name TEXT, status TEXT,
                  priority TEXT, collaborators TEXT, deadline TEXT,
                  notes TEXT, timestamp TEXT, type TEXT, created_at TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS commute_context
                 (id TEXT PRIMARY KEY, duration TEXT, end_location TEXT,
                  notes TEXT, start_location TEXT, timestamp TEXT,
                  traffic_condition TEXT, transport_mode TEXT, type TEXT,
                  created_at TEXT)''')
    
    conn.commit()
    conn.close()

# Initialize the database when the app starts
init_db()

def save_to_local_db(data_type, data):
    """Save data to local SQLite database"""
    conn = sqlite3.connect('local_data.db')
    c = conn.cursor()
    
    try:
        # Convert Firebase timestamps to strings
        if isinstance(data.get('timestamp'), datetime.datetime):
            data['timestamp'] = data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(data.get('created_at'), datetime.datetime):
            data['created_at'] = data['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            
        if data_type == 'query':
            c.execute('''INSERT OR REPLACE INTO queries 
                        (id, query, timestamp, source, processed, answer)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                     (str(data.get('id')), str(data.get('query')), str(data.get('timestamp')),
                      str(data.get('source')), bool(data.get('processed', False)),
                      json.dumps(data.get('answer', {}))))
        
        elif data_type == 'health_context':
            c.execute('''INSERT OR REPLACE INTO health_context 
                        (id, blood_sugar, created_at, exercise_minutes,
                         meal_type, medication, notes, timestamp, type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (str(data.get('id')), str(data.get('bloodSugar')), str(data.get('created_at')),
                      str(data.get('exerciseMinutes')), str(data.get('mealType')),
                      str(data.get('medication')), str(data.get('notes', '')), str(data.get('timestamp')),
                      str(data.get('type'))))
        
        elif data_type == 'work_context':
            c.execute('''INSERT OR REPLACE INTO work_context 
                        (id, task_name, status, priority, collaborators,
                         deadline, notes, timestamp, type, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (str(data.get('id')), str(data.get('taskName')), str(data.get('status')),
                      str(data.get('priority')), str(data.get('collaborators')),
                      str(data.get('deadline')), str(data.get('notes', '')), str(data.get('timestamp')),
                      str(data.get('type')), str(data.get('created_at'))))
        
        elif data_type == 'commute_context':
            c.execute('''INSERT OR REPLACE INTO commute_context 
                        (id, duration, end_location, notes, start_location,
                         timestamp, traffic_condition, transport_mode, type, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (str(data.get('id')), str(data.get('duration')), str(data.get('endLocation')),
                      str(data.get('notes', '')), str(data.get('startLocation')), str(data.get('timestamp')),
                      str(data.get('trafficCondition')), str(data.get('transportMode')),
                      str(data.get('type')), str(data.get('created_at'))))
        
        conn.commit()
    except Exception as e:
        print(f"Error saving to local database: {str(e)}")
    finally:
        conn.close()

def get_from_local_db(data_type, limit=1):
    """Retrieve data from local SQLite database"""
    conn = sqlite3.connect('local_data.db')
    c = conn.cursor()
    
    try:
        if data_type == 'query':
            c.execute('SELECT * FROM queries ORDER BY timestamp DESC LIMIT ?', (limit,))
        elif data_type == 'health_context':
            c.execute('SELECT * FROM health_context ORDER BY created_at DESC LIMIT ?', (limit,))
        elif data_type == 'work_context':
            c.execute('SELECT * FROM work_context ORDER BY created_at DESC LIMIT ?', (limit,))
        elif data_type == 'commute_context':
            c.execute('SELECT * FROM commute_context ORDER BY created_at DESC LIMIT ?', (limit,))
        
        rows = c.fetchall()
        if not rows:
            return None
        
        # Convert row to dictionary
        columns = [description[0] for description in c.description]
        data = dict(zip(columns, rows[0]))
        
        # Convert JSON strings back to objects
        if data_type == 'query' and data.get('answer'):
            data['answer'] = json.loads(data['answer'])
        
        return data
    except Exception as e:
        print(f"Error retrieving from local database: {str(e)}")
        return None
    finally:
        conn.close()

def check_existing_embedding(metadata):
    """Check if an embedding with the same metadata already exists"""
    try:
        # Get document from Pinecone
        doc_id = f"{metadata['type']}_{metadata['id']}"
        result = pinecone_service.fetch(doc_id)
        return result and result['vectors'] and doc_id in result['vectors']
    except Exception as e:
        print(f"Error checking existing embedding: {str(e)}")
        return False

def add_to_embeddings(text, metadata):
    """Add text to Pinecone embeddings if it doesn't already exist"""
    try:
        # Check if embedding already exists
        if check_existing_embedding(metadata):
            print(f"Embedding already exists for {metadata.get('type')} with ID {metadata.get('id')}")
            return False
            
        # Generate embedding
        embedding = model.encode(text).tolist()
        
        # Generate a unique document ID based on metadata
        doc_id = f"{metadata.get('type', 'doc')}_{metadata.get('id', 'unknown')}".replace(' ', '_').lower()
        
        # Add to Pinecone
        pinecone_service.upsert(doc_id, embedding, metadata)
        print(f"Added new embedding for {metadata.get('type')} with ID {metadata.get('id')}")
        return True
    except Exception as e:
        print(f"Error adding to embeddings: {str(e)}")
        return False

@app.route('/api/send-work-recommendation', methods=['POST'])
def send_work_recommendation():
    """Explicitly send a work recommendation to Firebase"""
    try:
        print("Received request to send work recommendation")
        data = request.get_json()
        print(f"Request data: {json.dumps(data, indent=2)}")
        
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Required fields
        required_fields = ['taskName', 'status', 'priority', 'recommendations']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Create recommendation data
        recommendation_data = {
            'taskName': data['taskName'],
            'status': data['status'],
            'priority': data['priority'],
            'recommendations': data['recommendations'],
            'collaborators': data.get('collaborators', ''),
            'deadline': data.get('deadline', ''),
            'notes': data.get('notes', ''),
            'work_context_id': data.get('work_context_id'),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': firestore.SERVER_TIMESTAMP
        }

        print(f"Preparing to save recommendation data: {json.dumps(recommendation_data, indent=2, default=str)}")

        # Add to Firebase
        recommendations_ref = db.collection('work_ai_recommendation')
        doc_ref = recommendations_ref.add(recommendation_data)
        doc_id = doc_ref[1].id
        print(f"Created new recommendation document with ID: {doc_id}")
        
        # Verify the save
        saved_doc = doc_ref[1].get()
        if not saved_doc.exists:
            raise Exception("Failed to save recommendation to Firebase")

        saved_data = saved_doc.to_dict()
        print(f"Verified saved document data: {json.dumps(saved_data, indent=2, default=str)}")

        return jsonify({
            "success": True,
            "message": "Recommendation sent to Firebase successfully",
            "recommendation_id": doc_id,
            "saved_data": saved_data
        })

    except Exception as e:
        error_msg = f"Error sending recommendation to Firebase: {str(e)}"
        print(f"Error details: {error_msg}")
        print(f"Stack trace: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

@app.route('/api/send-health-recommendation', methods=['POST'])
def send_health_recommendation():
    """Explicitly send a health recommendation to Firebase"""
    try:
        print("Received request to send health recommendation")
        data = request.get_json()
        print(f"Request data: {json.dumps(data, indent=2)}")
        
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Required fields
        required_fields = ['bloodSugar', 'exerciseMinutes', 'mealType', 'recommendations']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Create recommendation data
        recommendation_data = {
            'bloodSugar': data['bloodSugar'],
            'exerciseMinutes': data['exerciseMinutes'],
            'mealType': data['mealType'],
            'recommendations': data['recommendations'],
            'medication': data.get('medication', ''),
            'notes': data.get('notes', ''),
            'health_context_id': data.get('health_context_id'),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': firestore.SERVER_TIMESTAMP
        }

        print(f"Preparing to save health recommendation data: {json.dumps(recommendation_data, indent=2, default=str)}")

        # Add to Firebase
        recommendations_ref = db.collection('health_ai_recommendation')
        doc_ref = recommendations_ref.add(recommendation_data)
        doc_id = doc_ref[1].id
        print(f"Created new health recommendation document with ID: {doc_id}")
        
        # Verify the save
        saved_doc = doc_ref[1].get()
        if not saved_doc.exists:
            raise Exception("Failed to save health recommendation to Firebase")

        saved_data = saved_doc.to_dict()
        print(f"Verified saved document data: {json.dumps(saved_data, indent=2, default=str)}")

        return jsonify({
            "success": True,
            "message": "Health recommendation sent to Firebase successfully",
            "recommendation_id": doc_id,
            "saved_data": saved_data
        })

    except Exception as e:
        error_msg = f"Error sending health recommendation to Firebase: {str(e)}"
        print(f"Error details: {error_msg}")
        print(f"Stack trace: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

@app.route('/api/send-commute-recommendation', methods=['POST'])
def send_commute_recommendation():
    """Explicitly send a commute recommendation to Firebase"""
    try:
        print("Received request to send commute recommendation")
        data = request.get_json()
        print(f"Request data: {json.dumps(data, indent=2)}")
        
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Required fields
        required_fields = ['startLocation', 'endLocation', 'duration', 'recommendations']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Create recommendation data
        recommendation_data = {
            'startLocation': data['startLocation'],
            'endLocation': data['endLocation'],
            'duration': data['duration'],
            'recommendations': data['recommendations'],
            'trafficCondition': data.get('trafficCondition', ''),
            'transportMode': data.get('transportMode', ''),
            'notes': data.get('notes', ''),
            'commute_context_id': data.get('commute_context_id'),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': firestore.SERVER_TIMESTAMP
        }

        print(f"Preparing to save commute recommendation data: {json.dumps(recommendation_data, indent=2, default=str)}")

        # Add to Firebase
        recommendations_ref = db.collection('commute_ai_recommendation')
        doc_ref = recommendations_ref.add(recommendation_data)
        doc_id = doc_ref[1].id
        print(f"Created new commute recommendation document with ID: {doc_id}")
        
        # Verify the save
        saved_doc = doc_ref[1].get()
        if not saved_doc.exists:
            raise Exception("Failed to save commute recommendation to Firebase")

        saved_data = saved_doc.to_dict()
        print(f"Verified saved document data: {json.dumps(saved_data, indent=2, default=str)}")

        return jsonify({
            "success": True,
            "message": "Commute recommendation sent to Firebase successfully",
            "recommendation_id": doc_id,
            "saved_data": saved_data
        })

    except Exception as e:
        error_msg = f"Error sending commute recommendation to Firebase: {str(e)}"
        print(f"Error details: {error_msg}")
        print(f"Stack trace: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

@app.route('/api/create-context', methods=['POST'])
def create_context():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ['name', 'type', 'description']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Create context document
        context_data = {
            'name': data['name'],
            'type': data['type'],
            'description': data['description'],
            'tools': data.get('tools', []),
            'isPublic': data.get('isPublic', False),
            'created_at': firestore.SERVER_TIMESTAMP
        }

        # Add to Firestore
        doc_ref = db.collection('contexts').add(context_data)
        
        return jsonify({
            "success": True,
            "message": "Context created successfully",
            "context_id": doc_ref[1].id
        })

    except Exception as e:
        print(f"Error creating context: {str(e)}")
        return jsonify({"error": f"Error creating context: {str(e)}"}), 500

@app.route('/api/create-tool', methods=['POST'])
def create_tool():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Check required fields
        if not data.get('name') or not data.get('description'):
            return jsonify({"error": "Tool name and description are required"}), 400

        # Generate tool implementation using Gemini
        prompt = f"""Create a Python tool implementation based on this description:
"{data['description']}"

The implementation should include:
1. A clear docstring explaining the tool's purpose, parameters, and return value
2. Type hints for all parameters and return values
3. A Pydantic BaseModel for argument validation (if needed)
4. Proper error handling
5. The actual function implementation

Format the response as a JSON object with these fields:
{{
    "function": "the complete function implementation",
    "schema": "the Pydantic schema if needed, otherwise empty string",
    "return_direct": boolean indicating if the function returns a simple value
}}

Example format:
{{
    "function": "def calculate_average(numbers: List[float]) -> float:\\n    \\"\\"\\"Calculate the average of a list of numbers.\\n...\\"\\"\\"\\"",
    "schema": "class CalculateAverageInput(BaseModel):\\n    numbers: List[float]",
    "return_direct": true
}}"""

        # Get implementation from Gemini
        response = model_gemini.generate_content(prompt)
        if not response.text:
            return jsonify({"error": "Failed to generate tool implementation"}), 500

        try:
            # Parse the generated JSON
            import json
            implementation = json.loads(response.text)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract it from the text
            import re
            json_match = re.search(r'({[\s\S]*})', response.text)
            if json_match:
                implementation = json.loads(json_match.group(1))
            else:
                return jsonify({"error": "Invalid implementation format"}), 500

        # Create tool document
        tool_data = {
            'name': data['name'],
            'description': data['description'],
            'function': implementation['function'],
            'schema': implementation.get('schema', ''),
            'return_direct': implementation.get('return_direct', False),
            'created_at': firestore.SERVER_TIMESTAMP
        }

        # Add to Firestore
        doc_ref = db.collection('tools').add(tool_data)
        
        return jsonify({
            "success": True,
            "message": "Tool created successfully",
            "tool_id": doc_ref[1].id,
            "implementation": implementation
        })

    except Exception as e:
        print(f"Error creating tool: {str(e)}")
        return jsonify({"error": f"Error creating tool: {str(e)}"}), 500

@app.route('/api/tools')
def get_tools():
    """Get all tools from Firebase"""
    try:
        if db is None:
            return jsonify({"error": "Firebase not initialized"}), 500
            
        # Get all tools from Firebase
        tools_ref = db.collection('tools').order_by('created_at', direction=firestore.Query.DESCENDING)
        docs = tools_ref.get()
        
        tools = []
        for doc in docs:
            tool_data = doc.to_dict()
            tool_data['id'] = doc.id
            # Convert timestamp to string
            if 'created_at' in tool_data and tool_data['created_at']:
                tool_data['created_at'] = tool_data['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            tools.append(tool_data)
            
        return jsonify({"tools": tools})
            
    except Exception as e:
        print(f"Error fetching tools: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add after the init_db() function

def embed_context_in_rag(text, metadata):
    """Embed a context into the RAG system using Pinecone"""
    try:
        metadata = dict(metadata)
        metadata['text'] = text
        doc_id = f"{metadata['type']}_{metadata['id']}"
        embedding = model.encode(text).tolist()
        print(f"[DEBUG] Upserting to Pinecone: doc_id={doc_id}, metadata={metadata}")
        pinecone_service.upsert(doc_id, embedding, metadata)
        print(f"[DEBUG] Added/updated embedding for {doc_id}")
        return True
    except Exception as e:
        print(f"Error embedding context in RAG: {str(e)}")
        return False

def process_and_embed_firebase_contexts():
    """Fetch contexts from Firebase and embed them in the RAG system"""
    if not db:
        print("Firebase not initialized")
        return
    
    try:
        # Process health contexts
        health_contexts = db.collection('health_context').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).get()
        for doc in health_contexts:
            ctx_data = doc.to_dict()
            text = f"Health Context: Blood Sugar {ctx_data.get('bloodSugar')}, Exercise {ctx_data.get('exerciseMinutes')} mins"
            metadata = {
                'id': doc.id,
                'type': 'health_context',
                'source': 'Firebase',
                'created_at': ctx_data.get('created_at', firestore.SERVER_TIMESTAMP)
            }
            embed_context_in_rag(text, metadata)
        
        # Process health recommendations
        health_recs = db.collection('health_ai_recommendation').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).get()
        for doc in health_recs:
            rec_data = doc.to_dict()
            text = f"Health Recommendation: {', '.join(rec_data.get('recommendations', []))}"
            metadata = {
                'id': doc.id,
                'type': 'health_recommendation',
                'source': 'AI Recommendation',
                'created_at': rec_data.get('created_at', firestore.SERVER_TIMESTAMP)
            }
            embed_context_in_rag(text, metadata)
        
        # Process work contexts
        work_contexts = db.collection('work_context').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).get()
        for doc in work_contexts:
            ctx_data = doc.to_dict()
            text = f"Work Context: Task {ctx_data.get('taskName')}, Priority {ctx_data.get('priority')}, Status {ctx_data.get('status')}"
            metadata = {
                'id': doc.id,
                'type': 'work_context',
                'source': 'Firebase',
                'created_at': ctx_data.get('created_at', firestore.SERVER_TIMESTAMP)
            }
            embed_context_in_rag(text, metadata)
        
        # Process work recommendations
        work_recs = db.collection('work_ai_recommendation').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).get()
        for doc in work_recs:
            rec_data = doc.to_dict()
            text = f"Work Recommendation: {', '.join(rec_data.get('recommendations', []))}"
            metadata = {
                'id': doc.id,
                'type': 'work_recommendation',
                'source': 'AI Recommendation',
                'created_at': rec_data.get('created_at', firestore.SERVER_TIMESTAMP)
            }
            embed_context_in_rag(text, metadata)
        
        # Process commute contexts
        commute_contexts = db.collection('commute_context').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).get()
        for doc in commute_contexts:
            ctx_data = doc.to_dict()
            text = f"Commute Context: From {ctx_data.get('startLocation')} to {ctx_data.get('endLocation')} via {ctx_data.get('transportMode')}"
            metadata = {
                'id': doc.id,
                'type': 'commute_context',
                'source': 'Firebase',
                'created_at': ctx_data.get('created_at', firestore.SERVER_TIMESTAMP)
            }
            embed_context_in_rag(text, metadata)
        
        # Process commute recommendations
        commute_recs = db.collection('commute_ai_recommendation').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).get()
        for doc in commute_recs:
            rec_data = doc.to_dict()
            text = f"Commute Recommendation: {', '.join(rec_data.get('recommendations', []))}"
            metadata = {
                'id': doc.id,
                'type': 'commute_recommendation',
                'source': 'AI Recommendation',
                'created_at': rec_data.get('created_at', firestore.SERVER_TIMESTAMP)
            }
            embed_context_in_rag(text, metadata)
        
        # Process uploaded documents
        uploaded_docs = db.collection('uploaded_documents').order_by('created_at', direction=firestore.Query.DESCENDING).get()
        for doc in uploaded_docs:
            doc_data = doc.to_dict()
            storage_path = doc_data.get('storage_path')
            metadata = doc_data.get('metadata', {})
            metadata['id'] = doc.id
            metadata['type'] = 'document'
            metadata['source'] = 'Firebase'
            try:
                if not storage_path:
                    print(f"No storage_path for uploaded document {doc.id}")
                    continue
                bucket = storage.bucket()
                blob = bucket.blob(storage_path)
                file_bytes = blob.download_as_bytes()
                file_type = metadata.get('file_type', '').lower()
                text = ''
                if file_type == 'pdf':
                    try:
                        import io
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                        text = "".join([page.extract_text() or '' for page in pdf_reader.pages])
                    except Exception as e:
                        print(f"Error extracting PDF text for {doc.id}: {e}")
                        continue
                elif file_type == 'json':
                    try:
                        import json
                        text = json.loads(file_bytes.decode('utf-8'))
                        if isinstance(text, dict):
                            text = "\n".join(str(v) for v in text.values() if v is not None)
                        elif isinstance(text, list):
                            text = "\n".join(str(item) for item in text if item is not None)
                        else:
                            text = str(text)
                    except Exception as e:
                        print(f"Error extracting JSON text for {doc.id}: {e}")
                        continue
                else:  # txt or other
                    try:
                        text = file_bytes.decode('utf-8')
                    except Exception as e:
                        print(f"Error decoding text for {doc.id}: {e}")
                        continue
                embed_context_in_rag(text, metadata)
            except Exception as e:
                print(f"Error processing uploaded document {doc.id}: {e}")
            
    except Exception as e:
        print(f"Error processing and embedding Firebase contexts: {str(e)}")

@app.route('/contexts')
def contexts():
    return render_template('contexts.html')

@app.route('/api/contexts', methods=['GET'])
def get_contexts():
    try:
        all_contexts = []
        seen_ids = set()  # Track seen document IDs to prevent duplicates
        # Fetch from work_context collection
        work_docs = db.collection('work_context').get()
        for doc in work_docs:
            if doc.id in seen_ids:
                continue
            seen_ids.add(doc.id)
            data = doc.to_dict()
            doc_id = f"work_context_{doc.id}"
            pinecone_embedded = False
            fetch_result = pinecone_service.fetch(doc_id)
            if getattr(fetch_result, 'vectors', None) and doc_id in fetch_result.vectors:
                pinecone_embedded = True
            all_contexts.append({
                'id': doc.id,
                'document': f"Work Context: Task {data.get('taskName')}, Priority {data.get('priority')}, Status {data.get('status')}",
                'type': 'work_context',
                'source': 'Firebase',
                'created_at': data.get('created_at'),
                'metadata': {
                    'color': '#4285f4',  # Blue for work
                    'size': 10
                },
                'is_embedded': pinecone_embedded
            })
        # Fetch from commute_context collection
        commute_docs = db.collection('commute_context').get()
        for doc in commute_docs:
            if doc.id in seen_ids:
                continue
            seen_ids.add(doc.id)
            data = doc.to_dict()
            doc_id = f"commute_context_{doc.id}"
            pinecone_embedded = False
            fetch_result = pinecone_service.fetch(doc_id)
            if getattr(fetch_result, 'vectors', None) and doc_id in fetch_result.vectors:
                pinecone_embedded = True
            all_contexts.append({
                'id': doc.id,
                'document': f"Commute Context: From {data.get('startLocation')} to {data.get('endLocation')} via {data.get('transportMode')}",
                'type': 'commute_context',
                'source': 'Firebase',
                'created_at': data.get('created_at'),
                'metadata': {
                    'color': '#fbbc05',  # Yellow for commute
                    'size': 10
                },
                'is_embedded': pinecone_embedded
            })
        # Fetch from health_context collection
        health_docs = db.collection('health_context').get()
        for doc in health_docs:
            if doc.id in seen_ids:
                continue
            seen_ids.add(doc.id)
            data = doc.to_dict()
            doc_id = f"health_context_{doc.id}"
            pinecone_embedded = False
            fetch_result = pinecone_service.fetch(doc_id)
            if getattr(fetch_result, 'vectors', None) and doc_id in fetch_result.vectors:
                pinecone_embedded = True
            all_contexts.append({
                'id': doc.id,
                'document': f"Health Context: Blood Sugar {data.get('bloodSugar')}, Exercise {data.get('exerciseMinutes')}",
                'type': 'health_context',
                'source': 'Firebase',
                'created_at': data.get('created_at'),
                'metadata': {
                    'color': '#34a853',  # Green for health
                    'size': 10
                },
                'is_embedded': pinecone_embedded
            })
        # Fetch uploaded documents from Firestore
        uploaded_docs = db.collection('uploaded_documents').get()
        for doc in uploaded_docs:
            if doc.id in seen_ids:
                continue
            seen_ids.add(doc.id)
            doc_data = doc.to_dict()
            metadata = doc_data.get('metadata', {})
            file_name = doc_data.get('filename', 'Unknown file')
            display_text = f"File: {file_name} - {metadata.get('file_type', '')}"
            doc_id = f"document_{doc.id}"
            pinecone_embedded = False
            fetch_result = pinecone_service.fetch(doc_id)
            if getattr(fetch_result, 'vectors', None) and doc_id in fetch_result.vectors:
                pinecone_embedded = True
            all_contexts.append({
                'id': doc.id,
                'document': display_text,
                'type': 'document',
                'source': 'Firebase',
                'created_at': metadata.get('created_at', doc_data.get('created_at')),
                'metadata': {
                    'color': '#ea4335',  # Red for documents
                    'size': 10,
                    'file_type': metadata.get('file_type', ''),
                    'size_bytes': metadata.get('size', 0)
                },
                'is_embedded': pinecone_embedded
            })
        return jsonify({'contexts': all_contexts})
    except Exception as e:
        print(f"Error fetching contexts: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/embed-context/<context_id>', methods=['POST'])
def embed_context_api(context_id):
    try:
        # Try to find the context in Firebase
        # Check work_context
        doc = db.collection('work_context').document(context_id).get()
        if doc.exists:
            data = doc.to_dict()
            text = f"Work Context: Task {data.get('taskName')}, Priority {data.get('priority')}, Status {data.get('status')}"
            metadata = {
                'id': context_id,
                'type': 'work_context',
                'source': 'Firebase',
                'created_at': data.get('created_at')
            }
        else:
            # Check commute_context
            doc = db.collection('commute_context').document(context_id).get()
            if doc.exists:
                data = doc.to_dict()
                text = f"Commute Context: From {data.get('startLocation')} to {data.get('endLocation')} via {data.get('transportMode')}"
                metadata = {
                    'id': context_id,
                    'type': 'commute_context',
                    'source': 'Firebase',
                    'created_at': data.get('created_at')
                }
            else:
                # Check health_context
                doc = db.collection('health_context').document(context_id).get()
                if doc.exists:
                    data = doc.to_dict()
                    text = f"Health Context: Blood Sugar {data.get('bloodSugar')}, Exercise {data.get('exerciseMinutes')}"
                    metadata = {
                        'id': context_id,
                        'type': 'health_context',
                        'source': 'Firebase',
                        'created_at': data.get('created_at')
                    }
                else:
                    # Check uploaded_documents
                    doc = db.collection('uploaded_documents').document(context_id).get()
                    if doc.exists:
                        doc_data = doc.to_dict()
                        metadata = doc_data.get('metadata', {})
                        file_name = doc_data.get('filename', 'Unknown file')
                        text = f"File: {file_name} - {metadata.get('file_type', '')}"
                        metadata['id'] = context_id
                        metadata['type'] = 'document'
                        metadata['source'] = 'Firebase'
                        metadata['created_at'] = metadata.get('created_at', doc_data.get('created_at'))
                    else:
                        return jsonify({'success': False, 'error': 'Context not found'}), 404
        # Check if already embedded in Pinecone
        doc_id = f"{metadata['type']}_{metadata['id']}"
        fetch_result = pinecone_service.fetch(doc_id)
        if getattr(fetch_result, 'vectors', None) and doc_id in fetch_result.vectors:
            return jsonify({'success': True, 'already_embedded': True})
        # Not embedded, so embed now
        embedding = model.encode(text).tolist()
        pinecone_service.upsert(doc_id, embedding, metadata)
        return jsonify({'success': True, 'already_embedded': False})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/contexts', methods=['POST'])
def create_context_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        context_type = data.get('type')
        if not context_type:
            return jsonify({'error': 'Missing context type'}), 400

        # Log received data for debugging
        print(f"Received context data: {data}")

        # Handle work tracking context
        if context_type == 'work_tracking':
            required_fields = ['taskName', 'status', 'priority']
            missing_fields = [field for field in required_fields if not data.get(field)]
            
            if missing_fields:
                return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
            
            # Prepare context data
            context_data = {
                'type': context_type,
                'taskName': data.get('taskName'),
                'status': data.get('status'),
                'priority': data.get('priority'),
                'collaborators': data.get('collaborators', ''),
                'deadline': data.get('deadline', ''),
                'notes': data.get('notes', ''),
                'created_at': firestore.SERVER_TIMESTAMP,
                'timestamp': firestore.SERVER_TIMESTAMP
            }

            # Save to Firebase
            doc_ref = db.collection('work_context').document()
            doc_ref.set(context_data)
            
            # Get the actual document to return the server timestamp
            created_doc = doc_ref.get().to_dict()
            
            print(f"Successfully created work context with ID: {doc_ref.id}")
            return jsonify({
                'message': 'Context created successfully',
                'id': doc_ref.id,
                'data': created_doc
            }), 201

        # Handle commute tracking context
        elif context_type == 'commute_tracking':
            required_fields = ['startLocation', 'endLocation', 'duration']
            missing_fields = [field for field in required_fields if not data.get(field)]
            
            if missing_fields:
                return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
            
            context_data = {
                'type': context_type,
                'startLocation': data.get('startLocation'),
                'endLocation': data.get('endLocation'),
                'duration': data.get('duration'),
                'transportMode': data.get('transportMode', ''),
                'trafficCondition': data.get('trafficCondition', ''),
                'notes': data.get('notes', ''),
                'created_at': datetime.datetime.utcnow().isoformat() + 'Z',
                'timestamp': datetime.datetime.utcnow().strftime("%B %d, %Y at %I:%M:%S %p UTC%z")
            }
            
            doc_ref = db.collection('commute_context').document()
            doc_ref.set(context_data)
            return jsonify({
                'message': 'Context created successfully',
                'id': doc_ref.id,
                'data': context_data
            }), 201

        # Handle diabetes tracking context
        elif context_type == 'diabetes_tracking':
            required_fields = ['bloodSugar', 'exerciseMinutes', 'mealType']
            missing_fields = [field for field in required_fields if not data.get(field)]
            
            if missing_fields:
                return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
            
            context_data = {
                'type': context_type,
                'bloodSugar': data.get('bloodSugar'),
                'exerciseMinutes': data.get('exerciseMinutes'),
                'mealType': data.get('mealType'),
                'medication': data.get('medication', ''),
                'notes': data.get('notes', ''),
                'created_at': datetime.datetime.utcnow().isoformat() + 'Z',
                'timestamp': datetime.datetime.utcnow().strftime("%B %d, %Y at %I:%M:%S %p UTC%z")
            }
            
            doc_ref = db.collection('health_context').document()
            doc_ref.set(context_data)
            return jsonify({
                'message': 'Context created successfully',
                'id': doc_ref.id,
                'data': context_data
            }), 201

        else:
            return jsonify({'error': f'Invalid context type: {context_type}'}), 400

    except Exception as e:
        print(f"Error creating context: {str(e)}")
        traceback.print_exc()  # Print full stack trace
        return jsonify({'error': f'Failed to create context: {str(e)}'}), 500

@app.route('/api/contexts/<context_id>', methods=['PUT'])
def update_context(context_id):
    try:
        data = request.json
        if not data or 'type' not in data or 'text' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        if data['type'] not in ['health', 'work', 'commute']:
            return jsonify({'error': 'Invalid context type'}), 400
        
        if db is not None:
            collection_name = f"{data['type']}_contexts"
            
            # Update the context
            context_data = {
                'text': data['text'],
                'metadata': data.get('metadata', {}),
                'timestamp': datetime.now().isoformat(),
                'processed': False
            }
            
            # Update in Firebase
            doc_ref = db.collection(collection_name).document(context_id)
            doc_ref.update(context_data)
            
            return jsonify({
                'id': context_id,
                'type': data['type'],
                'text': data['text'],
                'metadata': data.get('metadata', {}),
                'timestamp': context_data['timestamp']
            })
        else:
            return jsonify({'error': 'Firebase database not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/contexts/<context_id>', methods=['DELETE'])
def delete_context(context_id):
    try:
        if db is not None:
            collections = ['health_context', 'work_context', 'commute_context']  # Changed from plural to singular
            
            for collection_name in collections:
                doc_ref = db.collection(collection_name).document(context_id)
                doc = doc_ref.get()
                if doc.exists:
                    doc_ref.delete()
                    return jsonify({'success': True})
            
            return jsonify({'error': 'Context not found'}), 404
        else:
            return jsonify({'error': 'Firebase database not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/document/<doc_id>', methods=['GET'])
def get_document(doc_id):
    try:
        # Get document from Pinecone
        result = pinecone_service.fetch(doc_id)
        
        if not result or not result['vectors']:
            return jsonify({"error": "Document not found"}), 404
            
        return jsonify({
            "id": doc_id,
            "document": result['vectors'][doc_id]['metadata'].get('text', ''),
            "metadata": result['vectors'][doc_id]['metadata']
        })
        
    except Exception as e:
        print(f"Error getting document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/document/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    try:
        # Check if document exists
        result = pinecone_service.fetch(doc_id)
        
        if not result or not result['vectors']:
            return jsonify({"error": "Document not found"}), 404
        
        # Delete from Pinecone
        pinecone_service.delete(doc_id)
        
        return jsonify({"success": True})
        
    except Exception as e:
        print(f"Error deleting document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/save-recommendation', methods=['POST'])
def save_recommendation():
    try:
        data = request.get_json()
        print('Received recommendation to save:', data)
        # Here you would add logic to save to Firebase
        # For now, just return success
        return jsonify({"success": True, "message": "Recommendation received."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/tools')
def tools_page():
    return render_template('tools.html')

@app.route('/api/activate-tool/<tool_type>', methods=['POST'])
def activate_tool(tool_type):
    try:
        # Here you would implement the actual tool activation logic
        # For now, we'll just return a success message
        response = {
            'status': 'success',
            'message': f'Tool {tool_type} activated successfully',
            'tool_type': tool_type
        }
        
        # Broadcast the tool activation to all connected WebSocket clients
        broadcast_to_websockets({
            'type': 'tool_activation',
            'tool_type': tool_type,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        return jsonify(response)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to activate tool: {str(e)}'
        }), 500

@app.route('/api/delete-embedding/<string:type>/<string:doc_id>', methods=['DELETE'])
def delete_embedding(type, doc_id):
    try:
        # Compose the Pinecone vector id as used in upserts
        pinecone_id = f"{type}_{doc_id}"
        pinecone_service.delete(pinecone_id)
        print(f"Deleted embedding from Pinecone: {pinecone_id}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting embedding: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 