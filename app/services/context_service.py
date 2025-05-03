from flask import jsonify
from datetime import datetime
import firebase_admin.firestore as firestore
from ..core.firebase import get_firebase_db
from ..core.chroma import get_chroma_collection

class ContextService:
    @staticmethod
    def get_content_key(doc_text, doc_type):
        """Create a unique content key for deduplication."""
        normalized_text = ' '.join(doc_text.lower().split())
        return f"{doc_type}:{normalized_text}"

    @staticmethod
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
        content_key = ContextService.get_content_key(display_text, doc_type)
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

    @staticmethod
    def get_contexts():
        """Get all contexts from both ChromaDB and Firebase."""
        try:
            all_contexts = []
            seen_ids = set()
            seen_content = set()
            
            # Fetch uploaded documents from ChromaDB
            collection = get_chroma_collection()
            if collection:
                try:
                    chroma_docs = collection.get()
                    if chroma_docs:
                        for i, doc in enumerate(chroma_docs['documents']):
                            doc_id = chroma_docs['ids'][i]
                            metadata = chroma_docs['metadatas'][i]
                            
                            # Create content key for deduplication
                            display_text = doc[:200] + '...' if len(doc) > 200 else doc
                            content_key = ContextService.get_content_key(display_text, metadata.get('type', 'document'))
                            
                            # Skip if we've seen this content or ID before
                            if doc_id in seen_ids or content_key in seen_content:
                                continue
                                
                            seen_ids.add(doc_id)
                            seen_content.add(content_key)
                            
                            # Format the document display text based on source
                            source = metadata.get('source', '')
                            if source.startswith('File:'):
                                display_text = f"Document: {source[6:]} - {doc[:100]}..."  # Show filename and preview
                            
                            all_contexts.append({
                                'id': doc_id,
                                'document': display_text,
                                'type': metadata.get('type', 'document'),
                                'source': source,
                                'created_at': metadata.get('created_at'),
                                'metadata': {
                                    'color': '#ea4335',  # Red for documents
                                    'size': 10,
                                    'file_type': metadata.get('file_type', ''),
                                    'size_bytes': metadata.get('size', 0)
                                }
                            })
                except Exception as e:
                    print(f"Error fetching ChromaDB documents: {str(e)}")
            
            # Fetch and process Firebase documents
            db = get_firebase_db()
            if db:
                try:
                    # Work contexts
                    work_docs = db.collection('work_context').get()
                    for doc in work_docs:
                        context = ContextService.process_firebase_doc(doc, 'work_context', '#4285f4', seen_ids, seen_content)
                        if context:
                            all_contexts.append(context)
                    
                    # Commute contexts
                    commute_docs = db.collection('commute_context').get()
                    for doc in commute_docs:
                        context = ContextService.process_firebase_doc(doc, 'commute_context', '#fbbc05', seen_ids, seen_content)
                        if context:
                            all_contexts.append(context)
                    
                    # Health contexts
                    health_docs = db.collection('health_context').get()
                    for doc in health_docs:
                        context = ContextService.process_firebase_doc(doc, 'health_context', '#34a853', seen_ids, seen_content)
                        if context:
                            all_contexts.append(context)
                except Exception as e:
                    print(f"Error fetching Firebase documents: {str(e)}")
            
            # Sort contexts by created_at timestamp
            all_contexts.sort(
                key=lambda x: x.get('created_at', ''),
                reverse=True
            )
            
            return jsonify({'contexts': all_contexts})
            
        except Exception as e:
            print(f"Error fetching contexts: {str(e)}")
            return jsonify({'error': str(e)}), 500

    @staticmethod
    def create_context(data):
        """Create a new context."""
        try:
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            context_type = data.get('type')
            if not context_type:
                return jsonify({'error': 'Missing context type'}), 400

            db = get_firebase_db()
            if not db:
                return jsonify({'error': 'Firebase not initialized'}), 500

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
                
                doc_ref = db.collection('work_context').document()
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
                
                doc_ref = db.collection('commute_context').document()
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
            return jsonify({'error': f'Failed to create context: {str(e)}'}), 500

    @staticmethod
    def update_context(context_id, data):
        """Update an existing context."""
        try:
            if not data or 'type' not in data or 'text' not in data:
                return jsonify({'error': 'Missing required fields'}), 400
            
            if data['type'] not in ['health', 'work', 'commute']:
                return jsonify({'error': 'Invalid context type'}), 400
            
            db = get_firebase_db()
            if not db:
                return jsonify({'error': 'Firebase database not initialized'}), 500
            
            collection_name = f"{data['type']}_context"
            
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
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @staticmethod
    def delete_context(context_id):
        """Delete a context."""
        try:
            db = get_firebase_db()
            if not db:
                return jsonify({'error': 'Firebase database not initialized'}), 500
            
            collections = ['health_context', 'work_context', 'commute_context']
            
            for collection_name in collections:
                doc_ref = db.collection(collection_name).document(context_id)
                doc = doc_ref.get()
                if doc.exists:
                    doc_ref.delete()
                    return jsonify({'success': True})
            
            return jsonify({'error': 'Context not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500 