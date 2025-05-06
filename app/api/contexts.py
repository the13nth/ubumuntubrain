from flask import Blueprint, request, jsonify
from datetime import datetime
import firebase_admin.firestore as firestore

from ..services.pinecone_service import pinecone_service
from ..services.firebase_service import firebase_service

# Initialize services
pinecone_service = pinecone_service()
firebase_service = firebase_service()

# Create blueprint
contexts_bp = Blueprint('contexts', __name__)

def get_content_key(doc_text, doc_type):
    """Create a unique content key for deduplication."""
    normalized_text = ' '.join(doc_text.lower().split())
    return f"{doc_type}:{normalized_text}"

def process_firebase_doc(doc, doc_type, color, seen_ids, seen_content):
    """Process a Firebase document into a standardized format."""
    if doc.id in seen_ids:
        return None
        
    data = doc.to_dict()
    
    # Create display text based on context type
    if doc_type == 'work_context':
        display_text = f"Work Context: Task {data.get('taskName')}, Priority {data.get('priority')}, Status {data.get('status')}"
    elif doc_type == 'commute_context':
        display_text = f"Commute Context: From {data.get('startLocation')} to {data.get('endLocation')} via {data.get('transportMode')}"
    elif doc_type == 'health_context':
        display_text = f"Health Context: Blood Sugar {data.get('bloodSugar')}, Exercise {data.get('exerciseMinutes')}"
    else:
        return None
        
    # Check for content duplication
    content_key = get_content_key(display_text, doc_type)
    if content_key in seen_content:
        return None
        
    seen_ids.add(doc.id)
    seen_content.add(content_key)
    
    return {
        'id': doc.id,
        'document': display_text,
        'type': doc_type,
        'source': 'Firebase',
        'created_at': data.get('created_at'),
        'metadata': {
            'color': color,
            'size': 10
        }
    }

@contexts_bp.route('/contexts', methods=['GET'])
def get_contexts():
    """Get all contexts from Pinecone and Firebase."""
    try:
        all_contexts = []
        seen_ids = set()
        
        # Fetch documents from Pinecone
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
                'source': 'Pinecone',
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
        print(f"Error fetching documents: {str(e)}")
        return jsonify({'error': str(e)}), 500

@contexts_bp.route('/', methods=['POST'])
def create_context():
    """Create a new context."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        context_type = data.get('type')
        if not context_type:
            return jsonify({'error': 'Missing context type'}), 400

        # Handle work tracking context
        if context_type == 'work_tracking':
            required_fields = ['taskName', 'status', 'priority']
            missing_fields = [field for field in required_fields if not data.get(field)]
            
            if missing_fields:
                return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
            
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
            
            doc_ref = firebase_service.db.collection('work_context').document()
            doc_ref.set(context_data)
            
            return jsonify({
                'message': 'Context created successfully',
                'id': doc_ref.id,
                'data': context_data
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
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'timestamp': datetime.utcnow().strftime("%B %d, %Y at %I:%M:%S %p UTC%z")
            }
            
            doc_ref = firebase_service.db.collection('commute_context').document()
            doc_ref.set(context_data)
            
            return jsonify({
                'message': 'Context created successfully',
                'id': doc_ref.id,
                'data': context_data
            }), 201

        # Handle health tracking context
        elif context_type == 'health_tracking':
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
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'timestamp': datetime.utcnow().strftime("%B %d, %Y at %I:%M:%S %p UTC%z")
            }
            
            doc_ref = firebase_service.db.collection('health_context').document()
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
        return jsonify({'error': f'Failed to create context: {str(e)}'}), 500

@contexts_bp.route('/<context_id>', methods=['PUT'])
def update_context(context_id):
    """Update an existing context."""
    try:
        data = request.json
        if not data or 'type' not in data or 'text' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        if data['type'] not in ['health', 'work', 'commute']:
            return jsonify({'error': 'Invalid context type'}), 400
        
        if firebase_service.db is not None:
            collection_name = f"{data['type']}_contexts"
            
            # Update the context
            context_data = {
                'text': data['text'],
                'metadata': data.get('metadata', {}),
                'timestamp': datetime.now().isoformat(),
                'processed': False
            }
            
            # Update in Firebase
            doc_ref = firebase_service.db.collection(collection_name).document(context_id)
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

@contexts_bp.route('/<context_id>', methods=['DELETE'])
def delete_context(context_id):
    """Delete a context."""
    try:
        if firebase_service.db is not None:
            collections = ['health_context', 'work_context', 'commute_context']
            
            for collection_name in collections:
                doc_ref = firebase_service.db.collection(collection_name).document(context_id)
                doc = doc_ref.get()
                if doc.exists:
                    doc_ref.delete()
                    return jsonify({'success': True})
            
            return jsonify({'error': 'Context not found'}), 404
        else:
            return jsonify({'error': 'Firebase database not initialized'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500 