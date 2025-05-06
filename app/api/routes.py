from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from datetime import datetime

from ..services.pinecone_service import pinecone_service
from ..services.firebase_service import firebase_service
from ..services.gemini_service import GeminiService
from ..utils.file_handler import allowed_file, process_file
from ..config.settings import Config

# Initialize services
gemini_service = GeminiService()

# Create blueprint
api = Blueprint('api', __name__)

@api.route('/')
def home():
    return render_template('query.html')

@api.route('/query')
def query_page():
    return render_template('query.html')

@api.route('/create')
def create_page():
    return render_template('create.html')

@api.route('/contexts')
def contexts_page():
    return render_template('contexts.html')

@api.route('/contexts', methods=['GET'])
def get_contexts():
    """Get all contexts from Firebase and Pinecone."""
    try:
        all_contexts = []
        seen_ids = set()
        
        # Get Pinecone documents
        pinecone_docs = pinecone_service.list_vectors()
        for doc_id in pinecone_docs:
            metadata = pinecone_service.get_metadata(doc_id)
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            all_contexts.append({
                'id': doc_id,
                'document': metadata.get('text', ''),
                'type': metadata.get('type', 'document'),
                'source': metadata.get('source', 'Pinecone'),
                'created_at': metadata.get('created_at'),
                'metadata': metadata,
                'is_embedded': True
            })
            
        # Get Firebase documents
        firebase_docs = firebase_service.get_all_documents()
        for doc in firebase_docs:
            if doc.id in seen_ids:
                continue
            seen_ids.add(doc.id)
            data = doc.to_dict()
            all_contexts.append({
                'id': doc.id,
                'document': data.get('text', ''),
                'type': data.get('type', 'document'),
                'source': 'Firebase',
                'created_at': data.get('created_at'),
                'metadata': data.get('metadata', {}),
                'is_embedded': False
            })
            
        return jsonify({'contexts': all_contexts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            
            # Ensure upload directory exists
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            
            # Save file
            file.save(filepath)
            
            # Process file and extract content
            content = process_file(file)
            
            # Save to Firebase
            firebase_service.save_context('document', {
                'content': content,
                'type': 'uploaded',
                'filename': filename
            })
            
            return jsonify({'message': 'File uploaded successfully'})
            
        return jsonify({'error': 'File type not allowed'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/query', methods=['POST'])
def process_query():
    """Process a user query."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
            
        query = data['query']
        
        # Search for relevant documents
        search_results = pinecone_service.search_vectors(query)
        
        # Generate response using Gemini
        context = None
        if search_results and search_results['documents']:
            context = "\n".join(search_results['documents'][0])
        
        response = gemini_service.generate_response(query, context)
        
        return jsonify({
            'response': response,
            'relevant_docs': search_results['documents'] if search_results else []
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500 