import PyPDF2
import json
from ..config.settings import Config

def allowed_file(filename):
    """Check if an uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def extract_text_from_pdf(file):
    """Extract text from a PDF file."""
    try:
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(file)
        
        # Extract text from all pages
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def extract_text_from_json(file):
    """Extract text from a JSON file."""
    try:
        # Load and parse JSON
        data = json.loads(file.read().decode('utf-8'))
        
        # Convert JSON to string, handling nested structures
        if isinstance(data, dict):
            # If it's a dictionary, extract all values
            text = "\n".join(str(v) for v in data.values() if v is not None)
        elif isinstance(data, list):
            # If it's a list, extract all items
            text = "\n".join(str(item) for item in data if item is not None)
        else:
            # If it's a simple value, convert to string
            text = str(data)
        
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from JSON: {str(e)}")

def process_file(file):
    """Process an uploaded file and extract its text content."""
    try:
        if file.filename.endswith('.pdf'):
            return extract_text_from_pdf(file)
        elif file.filename.endswith('.json'):
            return extract_text_from_json(file)
        else:  # .txt file
            return file.read().decode('utf-8')
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}") 