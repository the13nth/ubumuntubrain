import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime
from ..config.settings import Config
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirebaseService:
    """Service for managing Firebase operations."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirebaseService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.db = None
            self.storage = None
            self._initialized = True
    
    def initialize(self):
        """Initialize Firebase services."""
        if self.db is not None:
            return
            
        try:
            logger.info("Initializing Firebase...")
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                cred = credentials.Certificate('firebase-key.json')
                firebase_admin.initialize_app(cred, {
                    'storageBucket': 'ubumuntu-8d53c.appspot.com'
                })
            else:
                # Get the default app if it's already initialized
                app = firebase_admin.get_app()
            
            self.db = firestore.client()
            # Initialize storage as the storage module, not a bucket instance
            self.storage = storage
            logger.info("Firebase initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Firebase: {str(e)}")
            self.db = None
            self.storage = None
    
    def get_db(self):
        """Get the Firestore database client."""
        if not self.db:
            self.initialize()
        return self.db
    
    def get_storage(self):
        """Get the Firebase Storage bucket."""
        if not self.storage:
            self.initialize()
        return self.storage
    
    def save_query(self, query_text, query_type="user_query"):
        """Save a query to Firebase."""
        try:
            if not self.db:
                return None
                
            query_data = {
                'query': query_text,
                'type': query_type,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'created_at': datetime.utcnow().isoformat()
            }
            
            # Add to uqueries collection
            doc_ref = self.db.collection('uqueries').add(query_data)
            return doc_ref[1].id
            
        except Exception as e:
            print(f"Error saving query to Firebase: {str(e)}")
            return None
    
    def get_context(self, context_type, limit=10):
        """Get contexts from Firebase by type."""
        try:
            if not self.db:
                return []
                
            collection_name = f"{context_type}_context"
            docs = self.db.collection(collection_name)\
                        .order_by('created_at', direction=firestore.Query.DESCENDING)\
                        .limit(limit)\
                        .get()
            
            return [{'id': doc.id, **doc.to_dict()} for doc in docs]
            
        except Exception as e:
            print(f"Error getting {context_type} contexts: {str(e)}")
            return []
    
    def save_context(self, context_type, data):
        """Save a context to Firebase."""
        try:
            if not self.db:
                return None
                
            # Add timestamps
            data.update({
                'created_at': firestore.SERVER_TIMESTAMP,
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            
            # Save to appropriate collection
            collection_name = f"{context_type}_context"
            doc_ref = self.db.collection(collection_name).document()
            doc_ref.set(data)
            
            # Get the saved document
            saved_doc = doc_ref.get()
            return {'id': doc_ref.id, **saved_doc.to_dict()}
            
        except Exception as e:
            print(f"Error saving {context_type} context: {str(e)}")
            return None
    
    def save_recommendation(self, context_type, recommendations, context_data):
        """Save recommendations to Firebase."""
        try:
            if not self.db:
                return None
                
            # Prepare recommendation data
            rec_data = {
                'recommendations': recommendations,
                **context_data,
                f'{context_type}_context_id': context_data.get('id'),
                'timestamp': firestore.SERVER_TIMESTAMP,
                'created_at': firestore.SERVER_TIMESTAMP
            }
            
            # Save to appropriate collection
            collection_name = f"{context_type}_ai_recommendation"
            doc_ref = self.db.collection(collection_name).add(rec_data)
            return doc_ref[1].id
            
        except Exception as e:
            print(f"Error saving {context_type} recommendation: {str(e)}")
            return None
    
    def delete_context(self, context_type, context_id):
        """Delete a context from Firebase."""
        try:
            if not self.db:
                return False
                
            collection_name = f"{context_type}_context"
            self.db.collection(collection_name).document(context_id).delete()
            return True
            
        except Exception as e:
            print(f"Error deleting {context_type} context: {str(e)}")
            return False 
    
    def process_and_embed_firebase_contexts(self):
        """Fetch contexts from Firebase and embed them in the RAG system"""
        if not self.db:
            print("Firebase not initialized")
            return
        
        try:
            # Process health contexts
            health_contexts = self.db.collection('health_context').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).get()
            for doc in health_contexts:
                ctx_data = doc.to_dict()
                text = f"Health Context: Blood Sugar {ctx_data.get('bloodSugar')}, Exercise {ctx_data.get('exerciseMinutes')} mins"
                metadata = {
                    'id': doc.id,
                    'type': 'health_context',
                    'source': 'Firebase',
                    'created_at': ctx_data.get('created_at', firestore.SERVER_TIMESTAMP)
                }
                self.pinecone_service.upsert(metadata['id'], text, metadata)
            
            # Process health recommendations
            health_recs = self.db.collection('health_ai_recommendation').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).get()
            for doc in health_recs:
                rec_data = doc.to_dict()
                text = f"Health Recommendation: {', '.join(rec_data.get('recommendations', []))}"
                metadata = {
                    'id': doc.id,
                    'type': 'health_recommendation',
                    'source': 'AI Recommendation',
                    'created_at': rec_data.get('created_at', firestore.SERVER_TIMESTAMP)
                }
                self.pinecone_service.upsert(metadata['id'], text, metadata)
            
            # Process work contexts
            work_contexts = self.db.collection('work_context').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).get()
            for doc in work_contexts:
                ctx_data = doc.to_dict()
                text = f"Work Context: Task {ctx_data.get('taskName')}, Priority {ctx_data.get('priority')}, Status {ctx_data.get('status')}"
                metadata = {
                    'id': doc.id,
                    'type': 'work_context',
                    'source': 'Firebase',
                    'created_at': ctx_data.get('created_at', firestore.SERVER_TIMESTAMP)
                }
                self.pinecone_service.upsert(metadata['id'], text, metadata)
            
            # Process work recommendations
            work_recs = self.db.collection('work_ai_recommendation').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).get()
            for doc in work_recs:
                rec_data = doc.to_dict()
                text = f"Work Recommendation: {', '.join(rec_data.get('recommendations', []))}"
                metadata = {
                    'id': doc.id,
                    'type': 'work_recommendation',
                    'source': 'AI Recommendation',
                    'created_at': rec_data.get('created_at', firestore.SERVER_TIMESTAMP)
                }
                self.pinecone_service.upsert(metadata['id'], text, metadata)
            
            # Process commute contexts
            commute_contexts = self.db.collection('commute_context').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).get()
            for doc in commute_contexts:
                ctx_data = doc.to_dict()
                text = f"Commute Context: From {ctx_data.get('startLocation')} to {ctx_data.get('endLocation')} via {ctx_data.get('transportMode')}"
                metadata = {
                    'id': doc.id,
                    'type': 'commute_context',
                    'source': 'Firebase',
                    'created_at': ctx_data.get('created_at', firestore.SERVER_TIMESTAMP)
                }
                self.pinecone_service.upsert(metadata['id'], text, metadata)
            
            # Process commute recommendations
            commute_recs = self.db.collection('commute_ai_recommendation').order_by('created_at', direction=firestore.Query.DESCENDING).limit(5).get()
            for doc in commute_recs:
                rec_data = doc.to_dict()
                text = f"Commute Recommendation: {', '.join(rec_data.get('recommendations', []))}"
                metadata = {
                    'id': doc.id,
                    'type': 'commute_recommendation',
                    'source': 'AI Recommendation',
                    'created_at': rec_data.get('created_at', firestore.SERVER_TIMESTAMP)
                }
                self.pinecone_service.upsert(metadata['id'], text, metadata)
                
        except Exception as e:
            print(f"Error processing and embedding Firebase contexts: {str(e)}") 

    def add_document(self, text, metadata):
        """Add a document to both Firebase and Pinecone"""
        try:
            # Add to Pinecone
            self.pinecone_service.upsert(metadata['id'], text, metadata)
            return True
        except Exception as e:
            print(f"Error adding document: {str(e)}")
            return False

# Create singleton instance
firebase_service = FirebaseService() 