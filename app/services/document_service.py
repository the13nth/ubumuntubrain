import os
from flask import jsonify
from werkzeug.utils import secure_filename
import PyPDF2
import json
from app.services.firebase_service import firebase_service
import openpyxl
import csv

class DocumentService:
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json', 'xlsx', 'csv'}
    
    @staticmethod
    def get_document(doc_id, mode='detailed'):
        try:
            db = firebase_service.db
            # Try uploaded_documents first
            doc_ref = db.collection('uploaded_documents').document(doc_id)
            doc = doc_ref.get()
            if doc.exists:
                doc_data = doc.to_dict()
                if mode == 'quick':
                    return doc_data.get('summary') or doc_data.get('content') or doc_data.get('text') or doc_data.get('filename', '')
                else:
                    # Try to get full content
                    if doc_data.get('content'):
                        return doc_data['content']
                    # If not present, try to extract from storage
                    storage_path = doc_data.get('storage_path')
                    file_type = doc_data.get('metadata', {}).get('file_type', '').lower()
                    if storage_path and file_type:
                        try:
                            bucket = firebase_service.storage.bucket('ubumuntu-8d53c.firebasestorage.app')
                            blob = bucket.blob(storage_path)
                            file_bytes = blob.download_as_bytes()
                            import io
                            if file_type == 'pdf':
                                return DocumentService.extract_text_from_pdf(io.BytesIO(file_bytes))
                            elif file_type == 'json':
                                return DocumentService.extract_text_from_json(io.BytesIO(file_bytes))
                            elif file_type == 'xlsx':
                                import openpyxl
                                return DocumentService.extract_text_from_xlsx(io.BytesIO(file_bytes))
                            elif file_type == 'csv':
                                import csv
                                return file_bytes.decode('utf-8')
                            else:
                                return file_bytes.decode('utf-8', errors='ignore')
                        except Exception as e:
                            print(f"Error extracting full content from storage for {doc_id}: {e}")
                    # Fallback to text or filename
                    return doc_data.get('text') or doc_data.get('filename', '')
            # Fallback to documents collection
            doc_ref = db.collection('documents').document(doc_id)
            doc = doc_ref.get()
            if doc.exists:
                doc_data = doc.to_dict()
                return doc_data.get('content') or doc_data.get('text') or doc_data.get('filename', '')
            return ''
        except Exception as e:
            print(f"Error in get_document: {str(e)}")
            return ''
    
    @staticmethod
    def delete_document(doc_id):
        try:
            db = firebase_service.db
            doc_ref = db.collection('documents').document(doc_id)
            doc_ref.delete()
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
    def extract_text_from_xlsx(file):
        try:
            wb = openpyxl.load_workbook(file)
            text = ""
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    row_text = ' '.join([str(cell) for cell in row if cell is not None])
                    if row_text:
                        text += row_text + '\n'
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from XLSX: {str(e)}")
    
    @staticmethod
    def extract_text_from_csv(file):
        try:
            text = ""
            with open(file, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    row_text = ' '.join([str(cell) for cell in row if cell is not None])
                    if row_text:
                        text += row_text + '\n'
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from CSV: {str(e)}")
    
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
                elif file_ext == 'xlsx':
                    text = DocumentService.extract_text_from_xlsx(filepath)
                elif file_ext == 'csv':
                    text = DocumentService.extract_text_from_csv(filepath)
                else:  # txt files
                    text = f.read().decode('utf-8')
            
            # Clean up the uploaded file
            os.remove(filepath)
            
            return text
        else:
            raise Exception("Invalid file type") 