import warnings
# Filter out the specific FutureWarning from transformers
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers.utils.generic')

from flask import Flask, render_template, request, jsonify
import os
from flask_sock import Sock
import threading
from flask_cors import CORS
from dotenv import load_dotenv
import json

from app.services.context_service import ContextService
from app.services.document_service import DocumentService
from app.services.recommendation_service import RecommendationService
from app.services.query_service import QueryService
from app.services.visualization_service import VisualizationService
from app.core.database import init_db
from app.core.firebase import initialize_firebase
from app.core.chroma import initialize_chroma
from app.core.gemini import initialize_gemini

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Store WebSocket connections
ws_connections = set()
ws_lock = threading.Lock()

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize services
initialize_firebase()
initialize_chroma()
initialize_gemini()
init_db()

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

# API Routes
@app.route('/api/contexts', methods=['GET'])
def get_contexts():
    return ContextService.get_contexts()

@app.route('/api/contexts', methods=['POST'])
def create_context():
    return ContextService.create_context(request.json)

@app.route('/api/contexts/<context_id>', methods=['PUT'])
def update_context(context_id):
    return ContextService.update_context(context_id, request.json)

@app.route('/api/contexts/<context_id>', methods=['DELETE'])
def delete_context(context_id):
    return ContextService.delete_context(context_id)

@app.route('/api/document/<doc_id>', methods=['GET'])
def get_document(doc_id):
    return DocumentService.get_document(doc_id)

@app.route('/api/document/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    return DocumentService.delete_document(doc_id)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        text = DocumentService.process_uploaded_file(file, app.config['UPLOAD_FOLDER'])
        return jsonify({'text': text})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        context_type = data.get('context_type')
        result = QueryService.process_search_query(data['query'], context_type)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings-visualization')
def get_embeddings_visualization():
    return VisualizationService.get_embeddings_visualization()

@app.route('/api/save-recommendations', methods=['POST'])
def save_recommendations():
    return RecommendationService.save_recommendations(request.json, 'general')

@app.route('/api/save-work-recommendations', methods=['POST'])
def save_work_recommendations():
    return RecommendationService.save_recommendations(request.json, 'work')

@app.route('/api/save-health-recommendations', methods=['POST'])
def save_health_recommendations():
    return RecommendationService.save_recommendations(request.json, 'health')

@app.route('/api/save-commute-recommendations', methods=['POST'])
def save_commute_recommendations():
    return RecommendationService.save_recommendations(request.json, 'commute')

@app.route('/api/send-work-recommendation', methods=['POST'])
def send_work_recommendation():
    return RecommendationService.send_recommendation(request.json, 'work')

@app.route('/api/send-health-recommendation', methods=['POST'])
def send_health_recommendation():
    return RecommendationService.send_recommendation(request.json, 'health')

@app.route('/api/send-commute-recommendation', methods=['POST'])
def send_commute_recommendation():
    return RecommendationService.send_recommendation(request.json, 'commute')

# Frontend Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query')
def query_page():
    return render_template('query.html')

@app.route('/create')
def create_page():
    return render_template('create.html')

@app.route('/contexts')
def contexts():
    return render_template('contexts.html')

if __name__ == '__main__':
    app.run(debug=True) 