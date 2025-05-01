import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from ..config import Config

class FirebaseClient:
    def __init__(self):
        self.db = None
        self.initialize()
    
    def initialize(self):
        """Initialize Firebase connection"""
        try:
            cred = credentials.Certificate(Config.FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            print("Firebase initialized successfully")
        except Exception as e:
            print(f"Error initializing Firebase: {str(e)}")
            self.db = None
    
    def save_recommendation(self, collection_name, data):
        """Save recommendation to Firebase"""
        if not self.db:
            raise Exception("Firebase not initialized")
        
        try:
            # Add timestamps
            data.update({
                'timestamp': firestore.SERVER_TIMESTAMP,
                'created_at': firestore.SERVER_TIMESTAMP
            })
            
            # Add to Firebase
            doc_ref = self.db.collection(collection_name).add(data)
            return doc_ref[1].id
            
        except Exception as e:
            print(f"Error saving recommendation: {str(e)}")
            raise e
    
    def get_latest_context(self, collection_name, limit=1):
        """Get latest context from a collection"""
        if not self.db:
            raise Exception("Firebase not initialized")
            
        try:
            docs = self.db.collection(collection_name)\
                         .order_by('created_at', direction=firestore.Query.DESCENDING)\
                         .limit(limit)\
                         .get()
            
            if not docs:
                return None if limit == 1 else []
                
            if limit == 1:
                doc = next(iter(docs))
                data = doc.to_dict()
                data['id'] = doc.id
                return data
                
            return [{**doc.to_dict(), 'id': doc.id} for doc in docs]
            
        except Exception as e:
            print(f"Error getting context: {str(e)}")
            raise e
    
    def update_context(self, collection_name, context_id, data):
        """Update existing context"""
        if not self.db:
            raise Exception("Firebase not initialized")
            
        try:
            doc_ref = self.db.collection(collection_name).document(context_id)
            doc_ref.update(data)
            return True
            
        except Exception as e:
            print(f"Error updating context: {str(e)}")
            raise e
    
    def delete_context(self, context_id):
        """Delete context from all collections"""
        if not self.db:
            raise Exception("Firebase not initialized")
            
        try:
            collections = ['health_context', 'work_context', 'commute_context']
            
            for collection_name in collections:
                doc_ref = self.db.collection(collection_name).document(context_id)
                doc = doc_ref.get()
                if doc.exists:
                    doc_ref.delete()
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error deleting context: {str(e)}")
            raise e

# Create global Firebase instance
firebase = FirebaseClient() 