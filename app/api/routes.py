from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from datetime import datetime

from ..services.chroma_service import ChromaService
from ..services.firebase_service import FirebaseService
from ..services.gemini_service import GeminiService
from ..utils.file_handler import allowed_file, process_file
from ..config.settings import Config

# Initialize services
chroma_service = ChromaService()
firebase_service = FirebaseService()
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
    """Get all contexts from both ChromaDB and Firebase."""
    try:
        all_contexts = []
        seen_ids = set()
        seen_content = set()
        
        # Helper function to create a unique content key
        def get_content_key(doc_text, doc_type):
            normalized_text = ' '.join(doc_text.lower().split())
            return f"{doc_type}:{normalized_text}"
        
        # Get ChromaDB documents
        if chroma_service.collection:
            chroma_docs = chroma_service.get_all_documents()
            if chroma_docs:
                for i, doc in enumerate(chroma_docs['documents']):
                    doc_id = chroma_docs['ids'][i]
                    metadata = chroma_docs['metadatas'][i]
                    
                    content_key = get_content_key(doc, metadata.get('type', 'unknown'))
                    if doc_id not in seen_ids and content_key not in seen_content:
                        seen_ids.add(doc_id)
                        seen_content.add(content_key)
                        all_contexts.append({
                            'id': doc_id,
                            'content': doc,
                            **metadata
                        })
        
        # Get Firebase contexts
        firebase_contexts = firebase_service.get_context('document')
        for ctx in firebase_contexts:
            content_key = get_content_key(ctx.get('content', ''), ctx.get('type', 'unknown'))
            if ctx['id'] not in seen_ids and content_key not in seen_content:
                seen_ids.add(ctx['id'])
                seen_content.add(content_key)
                all_contexts.append(ctx)
        
        # Sort by created_at timestamp
        all_contexts.sort(
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )
        
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
            
            # Save to ChromaDB
            metadata = {
                'type': 'uploaded',
                'filename': filename,
                'created_at': datetime.utcnow().isoformat()
            }
            chroma_service.add_document(content, metadata)
            
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
        
        # Save query to Firebase
        firebase_service.save_query(query)
        
        # Search for relevant documents
        search_results = chroma_service.search_documents(query)
        
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