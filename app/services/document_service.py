import os
from flask import jsonify
from werkzeug.utils import secure_filename
import PyPDF2
import json
from ..core.firebase import get_firebase_db
from ..core.chroma import get_chroma_collection

class DocumentService:
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json'}
    
    @staticmethod
    def get_document(doc_id):
        try:
            db = get_firebase_db()
            doc_ref = db.collection('documents').document(doc_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                return jsonify({'error': 'Document not found'}), 404
            
            doc_data = doc.to_dict()
            return jsonify(doc_data)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @staticmethod
    def delete_document(doc_id):
        try:
            db = get_firebase_db()
            doc_ref = db.collection('documents').document(doc_id)
            doc_ref.delete()
            
            # Also remove from ChromaDB if needed
            collection = get_chroma_collection()
            try:
                collection.delete(ids=[doc_id])
            except:
                pass  # Document might not exist in ChromaDB
            
            return jsonify({'message': 'Document deleted successfully'})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @staticmethod
    def extract_text_from_pdf(file):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_json(file):
        try:
            data = json.load(file)
            if isinstance(data, dict):
                return json.dumps(data, indent=2)
            elif isinstance(data, list):
                return "\n".join(str(item) for item in data)
            else:
                return str(data)
        except Exception as e:
            raise Exception(f"Error extracting text from JSON: {str(e)}")
    
    @staticmethod
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in DocumentService.ALLOWED_EXTENSIONS
    
    @staticmethod
    def process_uploaded_file(file, upload_folder):
        if file and DocumentService.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            # Extract text based on file type
            file_ext = filename.rsplit('.', 1)[1].lower()
            with open(filepath, 'rb') as f:
                if file_ext == 'pdf':
                    text = DocumentService.extract_text_from_pdf(f)
                elif file_ext == 'json':
                    text = DocumentService.extract_text_from_json(f)
                else:  # txt files
                    text = f.read().decode('utf-8')
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return text
        else:
            raise Exception("Invalid file type") 