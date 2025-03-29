from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
import PyPDF2
import io
import google.generativeai as genai
from dotenv import load_dotenv
from flask_sock import Sock
import json
import threading
from functools import wraps

# Load environment variables
load_dotenv()

app = Flask(__name__)
sock = Sock(app)

# Store WebSocket connections
ws_connections = set()
ws_lock = threading.Lock()

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize ChromaDB
client = chromadb.Client(Settings(
    persist_directory="chroma_db",
    anonymized_telemetry=False
))

# Create or get the collection
collection = client.get_or_create_collection(
    name="text_embeddings",
    metadata={"hnsw:space": "cosine"}
)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 1024,
    "candidate_count": 1
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
model_gemini = genai.GenerativeModel(model_name="gemini-2.0-flash-lite",
                                   generation_config=generation_config,
                                   safety_settings=safety_settings)

def broadcast_to_websockets(message):
    """Broadcast a message to all connected WebSocket clients."""
    with ws_lock:
        dead_sockets = set()
        for ws in ws_connections:
            try:
                ws.send(json.dumps(message))
            except Exception:
                dead_sockets.add(ws)
        
        # Remove dead connections
        for dead_ws in dead_sockets:
            ws_connections.remove(dead_ws)

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.getenv('EXTERNAL_API_KEY'):
            return jsonify({"error": "Invalid API key"}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/external/query', methods=['POST'])
@require_api_key
def external_query():
    """External API endpoint for queries."""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"error": "Query is required"}), 400
        
        # Process the query using the existing search function
        result = process_search_query(data['query'])
        
        # Notify WebSocket clients about the API call
        broadcast_to_websockets({
            "type": "api_call",
            "query": data['query'],
            "query_result": result
        })
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@sock.route('/ws')
def handle_websocket(ws):
    """Handle WebSocket connections."""
    with ws_lock:
        ws_connections.add(ws)
    try:
        while True:
            # Keep the connection alive
            ws.receive()
    except Exception:
        with ws_lock:
            ws_connections.remove(ws)

def process_search_query(query):
    """Process a search query and return the results."""
    # Generate query embedding
    query_embedding = model.encode(query).tolist()
    
    # Search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=['embeddings', 'documents']
    )
    
    # Get all embeddings including the query for PCA
    all_data = collection.get(include=['embeddings'])
    if not all_data or 'embeddings' not in all_data or not all_data['embeddings']:
        return {
            "answer": "No documents found in the database.",
            "query_embedding_visualization": {"x": 0, "y": 0, "z": 0}
        }
    
    # Process embeddings and get visualization data
    visualization_data = process_embeddings_for_visualization(
        all_data['embeddings'],
        query_embedding,
        results
    )
    
    # Generate answer using Gemini
    answer = generate_answer(query, results)
    
    return {
        "answer": answer,
        "query_embedding_visualization": visualization_data["query_point"]
    }

def process_embeddings_for_visualization(embeddings, query_embedding, results):
    """Process embeddings for visualization."""
    all_embeddings = np.array(embeddings)
    combined_embeddings = np.vstack([all_embeddings, query_embedding])
    
    # Apply PCA
    n_samples = combined_embeddings.shape[0]
    n_features = combined_embeddings.shape[1]
    n_components = min(3, n_samples - 1, n_features)
    
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(combined_embeddings)
    
    # Get query point and pad with zeros if needed
    query_point = reduced_embeddings[-1]
    query_coords = np.zeros(3)
    for i in range(min(3, len(query_point))):
        query_coords[i] = query_point[i]
    
    return {
        "query_point": {
            "x": float(query_coords[0]),
            "y": float(query_coords[1]),
            "z": float(query_coords[2])
        }
    }

def generate_answer(query, results):
    """Generate an answer using Gemini."""
    if results.get('documents') and results['documents'][0]:
        prompt = create_rag_prompt(query, results['documents'][0])
        response = model_gemini.generate_content(prompt)
        return response.text if response.text else "I couldn't generate a response based on the context."
    return "I couldn't find any relevant information to answer your question."

# Update the existing search endpoint to use the new processing functions
@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        result = process_search_query(query)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in search: {str(e)}")
        return jsonify({"error": str(e)}), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

@app.route('/')
def home():
    return render_template('query.html')

@app.route('/query')
def query_page():
    return render_template('query.html')

@app.route('/create')
def create_page():
    return render_template('create.html')

@app.route('/api/create', methods=['POST'])
def create_embedding():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        text = data.get('text')
        if not text or not text.strip():
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        # Generate embedding
        embedding = model.encode(text.strip()).tolist()
        
        # Add to ChromaDB
        collection.add(
            embeddings=[embedding],
            documents=[text.strip()],
            ids=[f"text_{len(collection.get()['ids'])}"],
            metadatas=[{"source": "Manual Input"}]
        )
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Process the file based on its type
            if file.filename.endswith('.pdf'):
                text = extract_text_from_pdf(file)
            else:  # .txt file
                text = file.read().decode('utf-8')
            
            # Generate embedding
            embedding = model.encode(text).tolist()
            
            # Add to ChromaDB
            collection.add(
                embeddings=[embedding],
                documents=[text],
                ids=[f"doc_{len(collection.get()['ids'])}"],
                metadatas=[{"source": f"File: {file.filename}"}]
            )
            
            return jsonify({"success": True})
            
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    
    return jsonify({"success": False, "error": "Invalid file type"}), 400

@app.route('/api/data')
def get_data():
    try:
        # Get all data from ChromaDB with embeddings included
        data = collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        # If there's no data, return an empty list
        if not data or 'ids' not in data:
            return jsonify([])
        
        # Format the data for the table
        formatted_data = []
        for i in range(len(data['ids'])):
            formatted_data.append({
                'id': data['ids'][i],
                'document': data['documents'][i][:100] + '...' if len(data['documents'][i]) > 100 else data['documents'][i],
                'embedding_size': len(data['embeddings'][i]),  # This will now have the actual embedding size
                'source': data.get('metadatas', [{}])[i].get('source', 'Manual Input') if data.get('metadatas') else 'Manual Input'
            })
        
        return jsonify(formatted_data)
    except Exception as e:
        print(f"Error in get_data: {str(e)}")
        return jsonify([]), 500

@app.route('/api/embeddings-visualization')
def get_embeddings_visualization():
    try:
        # Get all data from ChromaDB with embeddings included
        data = collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        # If there's no data or no embeddings, return empty visualization
        if not data or 'embeddings' not in data or not data['embeddings']:
            return jsonify({
                "points": [],
                "variance_explained": [0, 0, 0]
            })
        
        # Convert embeddings to numpy array
        embeddings = np.array(data['embeddings'])
        
        # Determine number of components based on data size
        n_samples = embeddings.shape[0]
        n_features = embeddings.shape[1]
        n_components = min(3, n_samples - 1, n_features)
        
        if n_components < 1:
            return jsonify({
                "points": [],
                "variance_explained": [0, 0, 0]
            })
        
        # Apply PCA to reduce dimensions
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Create array for 3D coordinates
        coords_3d = np.zeros((reduced_embeddings.shape[0], 3))
        coords_3d[:, :n_components] = reduced_embeddings
        
        # Prepare the visualization data
        points = []
        for i in range(len(coords_3d)):
            points.append({
                'x': float(coords_3d[i, 0]),
                'y': float(coords_3d[i, 1]),
                'z': float(coords_3d[i, 2]),
                'text': data['documents'][i][:100] + '...' if len(data['documents'][i]) > 100 else data['documents'][i],
                'source': data.get('metadatas', [{}])[i].get('source', 'Manual Input') if data.get('metadatas') else 'Manual Input'
            })
        
        # Prepare variance explained (pad with zeros if needed)
        variance_explained = list(pca.explained_variance_ratio_)
        while len(variance_explained) < 3:
            variance_explained.append(0.0)
        
        return jsonify({
            "points": points,
            "variance_explained": variance_explained
        })
        
    except Exception as e:
        print(f"Error in get_embeddings_visualization: {str(e)}")
        return jsonify({
            "error": str(e),
            "points": [],
            "variance_explained": [0, 0, 0]
        }), 500

def create_rag_prompt(query, relevant_docs):
    # Format the context from relevant documents
    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(relevant_docs)])
    
    # Create the prompt
    prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. 
Your answers should be:
1. Accurate and based only on the provided context
2. Clear and well-structured
3. Include relevant quotes or references from the source documents when appropriate

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the above context. If the context doesn't contain enough information to answer the question fully, please state that explicitly."""

    return prompt

if __name__ == '__main__':
    app.run(debug=True) 