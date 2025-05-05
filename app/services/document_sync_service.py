import datetime
from app.core.chroma import chroma
from app.core.firebase import get_firebase_db
from app.services.document_service import DocumentService

class DocumentSyncService:
    @staticmethod
    def embed_and_save_document(file, upload_folder):
        """
        Extract text, embed in ChromaDB, and save to Firebase with embedded=True.
        Returns the Firebase doc ID and embedding status.
        """
        text = DocumentService.process_uploaded_file(file, upload_folder)
        # Generate a unique doc_id (could be filename+timestamp or Firebase ID after save)
        filename = file.filename
        doc_id = f"doc_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        metadata = {
            'type': 'document',
            'filename': filename,
            'created_at': datetime.datetime.now().isoformat(),
            'embedded': True,
        }
        # Embed in ChromaDB
        chroma.add_embedding(text, metadata, doc_id=doc_id)
        # Save to Firebase
        db = get_firebase_db()
        doc_ref = db.collection('documents').document(doc_id)
        doc_ref.set({
            'filename': filename,
            'content': text,
            'created_at': metadata['created_at'],
            'embedded': True,
        })
        return {'doc_id': doc_id, 'embedded': True}

    @staticmethod
    def sync_firebase_to_chroma():
        """
        For each document in Firebase, check if it's embedded in ChromaDB. If not, embed it and update Firebase.
        """
        db = get_firebase_db()
        docs = db.collection('documents').stream()
        synced = []
        for doc in docs:
            data = doc.to_dict()
            doc_id = doc.id
            embedded = data.get('embedded', False)
            text = data.get('content', '')
            filename = data.get('filename', '')
            if chroma.embedding_exists(doc_id):
                print(f"[SYNC] Skipping already embedded document: {doc_id} ({filename})")
                continue
            print(f"[SYNC] Embedding missing document: {doc_id} ({filename})")
            metadata = {
                'type': 'document',
                'filename': filename,
                'created_at': data.get('created_at', ''),
                'embedded': True,
            }
            chroma.add_embedding(text, metadata, doc_id=doc_id)
            db.collection('documents').document(doc_id).update({'embedded': True})
            synced.append(doc_id)
            print(f"[SYNC] Embedded and updated document: {doc_id} ({filename})")
        print(f"[SYNC] Sync complete. {len(synced)} documents embedded.")
        return synced 