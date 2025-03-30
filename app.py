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
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import datetime

# Load environment variables
load_dotenv()

# Initialize Firebase
try:
    cred = credentials.Certificate('firebase-key.json')
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully")
except Exception as e:
    print(f"Error initializing Firebase: {str(e)}")
    db = None

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)
sock = Sock(app)

# Store WebSocket connections
ws_connections = set()
ws_lock = threading.Lock()

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'json'}

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

# Try different model versions in case one fails
try:
    model_gemini = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                      generation_config=generation_config,
                                      safety_settings=safety_settings)
except Exception as e:
    print(f"Failed to initialize gemini-1.5-flash: {str(e)}")
    try:
        model_gemini = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                         generation_config=generation_config,
                                         safety_settings=safety_settings)
    except Exception as e:
        print(f"Failed to initialize gemini-1.0-pro: {str(e)}")
        try:
            model_gemini = genai.GenerativeModel(model_name="gemini-pro",
                                             generation_config=generation_config,
                                             safety_settings=safety_settings)
        except Exception as e:
            print(f"Failed to initialize gemini-pro: {str(e)}")
            model_gemini = None  # Will use fallback responses

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
        try:
            result = process_search_query(data['query'])
            
            # Notify WebSocket clients about the API call
            broadcast_to_websockets({
                "type": "api_call",
                "query": data['query'],
                "query_result": result
            })
            
            return jsonify(result)
            
        except Exception as e:
            error_message = str(e)
            print(f"Error processing query: {error_message}")
            
            # Send error notification to WebSocket clients
            broadcast_to_websockets({
                "type": "api_call_error",
                "query": data['query'],
                "error": error_message
            })
            
            return jsonify({
                "error": error_message,
                "answer": f"Error processing query: {error_message}",
                "query_embedding_visualization": {"x": 0, "y": 0, "z": 0}
            }), 500
    
    except Exception as e:
        error_message = str(e)
        print(f"External API error: {error_message}")
        return jsonify({"error": error_message}), 500

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
        include=['embeddings', 'documents', 'metadatas']
    )
    
    # Make sure we have results
    if not results or 'documents' not in results or not results['documents'] or not results['documents'][0]:
        return {
            "answer": "No documents found in the database.",
            "query_embedding_visualization": {"x": 0, "y": 0, "z": 0}
        }
    
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
    answer = generate_answer(query, results['documents'][0])
    
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

def generate_answer(query, relevant_docs):
    """Generate an answer using Gemini."""
    # Check for widescreen optimization specifically
    if "optimize" in query.lower() and "widescreen" in query.lower():
        # Look for the template in relevant docs
        for doc in relevant_docs:
            if isinstance(doc, str) and "layoutQueries" in doc and "widescreen" in doc:
                try:
                    import json
                    import re
                    
                    # Try to parse as JSON
                    json_data = json.loads(doc)
                    if isinstance(json_data, dict) and "layoutQueries" in json_data and "widescreen" in json_data["layoutQueries"]:
                        # Return the formatted layout JSON directly
                        widescreen_data = json_data["layoutQueries"]["widescreen"]
                        return """```json
{
  "cols": 4,
  "rows": 3,
  "rowHeight": 120,
  "margin": [15, 15]
}
```"""
                except Exception as e:
                    print(f"Error extracting widescreen template: {str(e)}")
    
    if relevant_docs:
        prompt = create_rag_prompt(query, relevant_docs)
        
        # If model isn't available, provide fallback response
        if model_gemini is None:
            return f"Found {len(relevant_docs)} relevant documents but AI generation is unavailable. Please check your API key."
        
        try:
            response = model_gemini.generate_content(prompt)
            return response.text if hasattr(response, 'text') and response.text else "I couldn't generate a response based on the context."
        except Exception as e:
            print(f"Error generating content: {str(e)}")
            # Return a helpful error message with retrieved document snippets
            doc_snippets = [doc[:100] + "..." if len(doc) > 100 else doc for doc in relevant_docs[:2]]
            snippets_text = "\n\n".join([f"Document snippet {i+1}: {snippet}" for i, snippet in enumerate(doc_snippets)])
            return f"Error generating AI response. Found {len(relevant_docs)} relevant documents. Here are some snippets:\n\n{snippets_text}"
    
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

def extract_text_from_json(json_file):
    """Extract text content from JSON file."""
    try:
        import json
        data = json.load(json_file)
        # Convert the JSON to a formatted string representation
        text = json.dumps(data, indent=2)
        return text
    except Exception as e:
        print(f"Error extracting text from JSON: {str(e)}")
        return f"Error parsing JSON: {str(e)}"

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
            elif file.filename.endswith('.json'):
                text = extract_text_from_json(file)
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
    # Check if this is a dashboard layout optimization request
    if "optimize" in query.lower() and ("display" in query.lower() or "layout" in query.lower() or "dashboard" in query.lower()):
        # For dashboard layout optimization requests
        dashboard_config = None
        
        # Try to extract dashboard configuration from relevant documents
        for doc in relevant_docs:
            if "cols" in doc and "rows" in doc and "rowHeight" in doc:
                try:
                    import json
                    import re
                    
                    # Extract JSON-like configuration using regex
                    config_match = re.search(r'({[^}]*"cols"[^}]*"rows"[^}]*})', doc)
                    if config_match:
                        config_text = config_match.group(1)
                        # Clean up potential issues with the extracted JSON
                        config_text = re.sub(r',\s*}', '}', config_text)
                        dashboard_config = json.loads(config_text)
                        break
                except Exception as e:
                    print(f"Error parsing dashboard config: {str(e)}")
        
        # If no configuration found, use defaults
        if not dashboard_config:
            dashboard_config = {
                "cols": 2,
                "rows": 2,
                "rowHeight": 200,
                "margin": [10, 10],
            }
        
        # Create specialized prompt for layout optimization
        prompt = f"""You are a UI layout assistant. Given this dashboard configuration:
      - Current columns: {dashboard_config.get('cols', 2)}
      - Current rows: {dashboard_config.get('rows', 2)}
      - Current row height: {dashboard_config.get('rowHeight', 200)}px
      - Current margin: {dashboard_config.get('margin', [10, 10])}

      User request: "{query}"

      Based on the user's request, return a response that includes:
      1. A layout configuration JSON object
      2. Health information for the user's diabetes monitoring

      The layout configuration must be a valid JSON object with these properties:
      {{
        "cols": (number between 1-12),
        "rows": (number between 1-20),
        "rowHeight": (number between 40-200),
        "margin": [number, number]
      }}

      The health information should be included in a "healthContext" object with this structure:
      {{
        "healthContext": {{
          "diabetesInfo": {{
            "lastReading": {{
              "glucoseLevel": (number),
              "timestamp": (ISO date string),
              "status": (one of: "Normal", "Slightly Elevated", "Elevated", "Low")
            }},
            "dailyStats": {{
              "averageGlucose": (number),
              "readings": (number),
              "inRange": (number),
              "high": (number),
              "low": (number)
            }},
            "medications": [
              {{
                "name": (string),
                "dosage": (string),
                "frequency": (string),
                "lastTaken": (ISO date string)
              }}
            ],
            "recommendations": [
              (string array of 2-3 recommendations)
            ]
          }}
        }}
      }}

      Format your answer as a code block with both the layout configuration and health context:
      ```json
      {{
        "response": {{
          "timestamp": "2024-03-20T14:30:00Z",
          "raw": "```json\\n{{\\n  \\"cols\\": 4,\\n  \\"rows\\": 3,\\n  \\"rowHeight\\": 120,\\n  \\"margin\\": [15, 15]\\n}}\\n```\\n",
          "parsed": {{
            "cols": 4,
            "rows": 3,
            "rowHeight": 120,
            "margin": [15, 15]
          }},
          "validated": {{
            "cols": 4,
            "rows": 3,
            "rowHeight": 120,
            "margin": [15, 15]
          }}
        }},
        "result": {{
          "finalConfig": {{
            "cols": 4,
            "rows": 3,
            "rowHeight": 120,
            "margin": [15, 15]
          }},
          "changes": {{
            "colsChanged": true,
            "rowsChanged": true,
            "rowHeightChanged": true,
            "marginChanged": true
          }}
        }},
        "healthContext": {{
          "diabetesInfo": {{
            "lastReading": {{
              "glucoseLevel": 110,
              "timestamp": "2024-03-20T14:30:00Z",
              "status": "Normal"
            }},
            "dailyStats": {{
              "averageGlucose": 112,
              "readings": 8,
              "inRange": 8,
              "high": 0,
              "low": 0
            }},
            "medications": [
              {{
                "name": "Metformin",
                "dosage": "500mg",
                "frequency": "Twice daily",
                "lastTaken": "2024-03-20T08:00:00Z"
              }}
            ],
            "recommendations": [
              "Schedule is on track",
              "Consider a short walk after next meal",
              "Next reading due before dinner"
            ]
          }}
        }}
      }}
      ```
      """
        
        return prompt
    
    # For standard RAG queries
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

@app.route('/firebase-query')
def fetch_firebase_query():
    """Fetch the latest query from Firebase and display UI to submit it to RAG"""
    try:
        if db is None:
            return render_template('firebase_error.html', error="Firebase not initialized")
        
        # Get the latest query from Firebase
        queries_ref = db.collection('rag_queries')
        query = queries_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).get()
        
        latest_query = None
        for doc in query:
            latest_query = doc.to_dict()
            latest_query['id'] = doc.id
            break
        
        return render_template('firebase_query.html', query=latest_query)
    
    except Exception as e:
        print(f"Error fetching from Firebase: {str(e)}")
        return render_template('firebase_error.html', error=str(e))

@app.route('/submit-firebase-query', methods=['POST'])
def submit_firebase_query():
    """Submit a query from Firebase to the RAG system"""
    try:
        query_text = request.form.get('query')
        if not query_text:
            return jsonify({"error": "No query provided"}), 400
        
        # Process the query
        result = process_search_query(query_text)
        
        # Update the Firebase document with the result
        if db is not None:
            query_id = request.form.get('query_id')
            if query_id:
                db.collection('rag_queries').document(query_id).update({
                    'answer': result['answer'],
                    'processed': True,
                    'processed_timestamp': firestore.SERVER_TIMESTAMP
                })
        
        return render_template('firebase_result.html', result=result, query=query_text)
    
    except Exception as e:
        print(f"Error processing Firebase query: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/latest-firebase-query')
def latest_firebase_query():
    """API endpoint to fetch the latest query from Firebase"""
    try:
        if db is None:
            return jsonify({"error": "Firebase not initialized"}), 500
        
        # Get the latest query from Firebase
        queries_ref = db.collection('rag_queries')
        query = queries_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).get()
        
        latest_query = None
        for doc in query:
            query_data = doc.to_dict()
            # Convert timestamp to string for JSON serialization if it exists
            if 'timestamp' in query_data and query_data['timestamp']:
                query_data['timestamp'] = query_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            query_data['id'] = doc.id
            latest_query = query_data
            break
        
        return jsonify({"query": latest_query})
    
    except Exception as e:
        print(f"Error fetching from Firebase: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/latest-health-context')
def latest_health_context():
    """API endpoint to fetch the latest health context from Firebase"""
    try:
        if db is None:
            return jsonify({"error": "Firebase not initialized"}), 500
            
        # Get the latest health context from Firebase
        health_ref = db.collection('health_context')
        readings = health_ref.order_by('created_at', direction=firestore.Query.DESCENDING).limit(1).get()
        
        latest_reading = None
        for doc in readings:
            health_data = {
                'bloodSugar': str(doc.get('bloodSugar')),  # ensure string
                'created_at': doc.get('created_at'),  # already string
                'exerciseMinutes': str(doc.get('exerciseMinutes')),  # ensure string
                'mealType': doc.get('mealType'),  # already string
                'medication': doc.get('medication'),  # already string
                'notes': doc.get('notes'),  # already string
                'timestamp': doc.get('timestamp'),  # firestore timestamp
                'type': doc.get('type')  # already string
            }
            latest_reading = health_data
            break
            
        return jsonify({"health_context": latest_reading})
        
    except Exception as e:
        print(f"Error fetching health context: {str(e)}")
        return jsonify({"error": f"Error fetching health context: {str(e)}"}), 500

def generate_layout_response(doc_ref, dashboard_context, raw_response, parsed_json):
    """Helper function to generate formatted layout response and update Firebase"""
    try:
        from datetime import datetime
        import json
        
        # Validate the parsed JSON
        validated = {
            "cols": min(max(1, parsed_json.get('cols', 2)), 12),
            "rows": min(max(1, parsed_json.get('rows', 2)), 20),
            "rowHeight": min(max(40, parsed_json.get('rowHeight', 200)), 200),
            "margin": parsed_json.get('margin', [10, 10])
        }
        
        # Create the final config by merging with original config
        original_config = dashboard_context.get('originalConfig', {})
        final_config = {**original_config}
        for key in validated:
            final_config[key] = validated[key]
        
        # Calculate changes
        changes = {
            "colsChanged": original_config.get('cols') != final_config.get('cols'),
            "rowsChanged": original_config.get('rows') != final_config.get('rows'),
            "rowHeightChanged": original_config.get('rowHeight') != final_config.get('rowHeight'),
            "marginChanged": original_config.get('margin') != final_config.get('margin')
        }
        
        # Format the complete response structure
        formatted_response = {
            "response": {
                "timestamp": datetime.now().isoformat(),
                "raw": raw_response,
                "parsed": parsed_json,
                "validated": validated
            },
            "result": {
                "finalConfig": final_config,
                "changes": changes
            }
        }
        
        # Update Firestore with the formatted response
        doc_ref.update({
            'answer': formatted_response,
            'processed': True,
            'processed_timestamp': firestore.SERVER_TIMESTAMP
        })
        
        # Return the formatted response
        return jsonify(formatted_response)
    except Exception as e:
        print(f"Error in generate_layout_response: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/process-firebase-query', methods=['POST'])
def process_firebase_query_api():
    """API endpoint to process a query from Firebase"""
    try:
        query_text = request.form.get('query')
        if not query_text:
            return jsonify({"error": "No query provided"}), 400
        
        # Extract query ID and prepare to get dashboard context
        query_id = request.form.get('query_id')
        dashboard_context = None
        
        # Get dashboard context from Firebase if available
        if db is not None and query_id:
            try:
                doc_ref = db.collection('rag_queries').document(query_id)
                doc = doc_ref.get()
                if doc.exists:
                    query_data = doc.to_dict()
                    if 'dashboardContext' in query_data:
                        dashboard_context = query_data['dashboardContext']
            except Exception as e:
                print(f"Error fetching Firebase document: {str(e)}")
        
        # Direct matching for common layout queries
        layout_config = None
        query_lower = query_text.lower()
        
        # Standard layout optimization
        if "optimize" in query_lower and "widescreen" in query_lower:
            layout_config = {
                "cols": 4,
                "rows": 3,
                "rowHeight": 120,
                "margin": [15, 15]
            }
        # Make cards bigger
        elif "bigger" in query_lower and ("card" in query_lower or "layout" in query_lower):
            # For bigger cards, reduce columns, increase row height
            layout_config = {
                "cols": 2,  # Smaller number of columns makes cards wider
                "rows": 4,
                "rowHeight": 180,  # Taller rows
                "margin": [15, 15]  # Slightly bigger margins
            }
        # Compact layout
        elif "compact" in query_lower or "smaller" in query_lower:
            layout_config = {
                "cols": 6,  # More columns for compact view
                "rows": 6,
                "rowHeight": 80,  # Smaller height
                "margin": [5, 5]  # Smaller margins
            }
        # Increase spacing
        elif "spacing" in query_lower or "space" in query_lower:
            layout_config = {
                "cols": 3,  # Fewer columns to allow more space
                "rows": 3,
                "rowHeight": 150,
                "margin": [20, 20]  # Larger margins for more space
            }
        # Square grid
        elif "square" in query_lower:
            layout_config = {
                "cols": 3,
                "rows": 3,  # Same number of rows as columns
                "rowHeight": 150,
                "margin": [10, 10]
            }
        # Custom grid size match
        elif "grid" in query_lower or "arrange" in query_lower:
            import re
            grid_match = re.search(r'(\d+)x(\d+)', query_lower)
            
            if grid_match:
                try:
                    cols = int(grid_match.group(1))
                    rows = int(grid_match.group(2))
                    # Ensure reasonable bounds
                    cols = min(max(1, cols), 12)
                    rows = min(max(1, rows), 20)
                    
                    layout_config = {
                        "cols": cols,
                        "rows": rows,
                        "rowHeight": 120,
                        "margin": [10, 10]
                    }
                except:
                    pass
        
        # Category matching
        category = None
        if "social" in query_lower or "social media" in query_lower:
            category = "social"
        elif "work" in query_lower:
            category = "work"
        elif "financial" in query_lower or "finance" in query_lower:
            category = "financial"
        elif "entertainment" in query_lower:
            category = "entertainment"
        elif "utility" in query_lower:
            category = "utility"
        elif "transportation" in query_lower:
            category = "transportation"
        elif "news" in query_lower:
            category = "news"
        elif "all" in query_lower and "app" in query_lower:
            category = "all"
        
        # If we have a layout config or category, generate a direct response
        if layout_config or category:
            from datetime import datetime
            import json
            
            if layout_config:
                # Format layout config as JSON
                raw_response = f"""```json
{json.dumps(layout_config, indent=2)}
```"""
                
                if dashboard_context and db is not None and query_id:
                    try:
                        doc_ref = db.collection('rag_queries').document(query_id)
                        # Format response and update Firebase
                        response_data = generate_layout_response(
                            doc_ref, 
                            dashboard_context, 
                            raw_response, 
                            layout_config
                        )
                        
                        # Save the raw response to rag_responses collection
                        if db is not None:
                            # Save only the clean JSON object, not the markdown code block
                            save_response_to_firebase(query_text, json.dumps(layout_config, indent=2), "layout", query_id)
                        
                        return response_data
                    except Exception as e:
                        print(f"Error processing layout response: {str(e)}")
            
            if category and db is not None and query_id:
                try:
                    # Create standardized category response format
                    response_data = {
                        "success": True,
                        "answer": f"Processed query: \"{query_text}\" for category: {category}",
                        "query_embedding_visualization": {
                            "x": 0.5,  # Default placeholder values
                            "y": 0.5, 
                            "z": 0.5
                        },
                        "category": category
                    }
                    
                    # Update the Firebase document with consistent format
                    doc_ref = db.collection('rag_queries').document(query_id)
                    doc_ref.update({
                        'answer': response_data,
                        'processed': True,
                        'processed_timestamp': firestore.SERVER_TIMESTAMP
                    })
                    
                    # Save the raw response to rag_responses collection
                    if db is not None:
                        save_response_to_firebase(query_text, json.dumps(response_data), "category", query_id)
                    
                    return jsonify(response_data)
                except Exception as e:
                    print(f"Error processing category response: {str(e)}")
        
        # If not a standard query, process using the regular RAG flow
        # Add dashboard context temporarily to ChromaDB if available
        temp_id = None
        if dashboard_context:
            try:
                import json
                dashboard_text = f"""Dashboard Configuration:
                cols: {dashboard_context.get('originalConfig', {}).get('cols', 2)}
                rows: {dashboard_context.get('originalConfig', {}).get('rows', 2)}
                rowHeight: {dashboard_context.get('originalConfig', {}).get('rowHeight', 200)}
                margin: {dashboard_context.get('originalConfig', {}).get('margin', [10, 10])}
                numApps: {dashboard_context.get('originalConfig', {}).get('numApps', 4)}
                activeCategory: {dashboard_context.get('originalConfig', {}).get('activeCategory', 'default')}
                
                Raw config: {json.dumps(dashboard_context.get('originalConfig', {}))}
                """
                
                # Add to ChromaDB temporarily for this query
                embedding = model.encode(dashboard_text).tolist()
                temp_id = f"temp_dashboard_{query_id}"
                
                # Check if we already have this temp document
                try:
                    collection.get(ids=[temp_id])
                    # If it exists, delete it first
                    collection.delete(ids=[temp_id])
                except:
                    pass
                    
                # Add the temporary dashboard context
                collection.add(
                    embeddings=[embedding],
                    documents=[dashboard_text],
                    ids=[temp_id],
                    metadatas=[{"source": "Firebase Dashboard Context", "temp": True}]
                )
                print(f"Added temporary dashboard context document with ID: {temp_id}")
            except Exception as e:
                print(f"Error adding dashboard context to ChromaDB: {str(e)}")
        
        # Process the query using standard RAG
        result = process_search_query(query_text)
        
        # Clean up temporary collection entry if it exists
        if temp_id:
            try:
                collection.delete(ids=[temp_id])
                print(f"Deleted temporary dashboard context document with ID: {temp_id}")
            except Exception as e:
                print(f"Error deleting temporary dashboard context: {str(e)}")
        
        # Check for layout patterns in the result and format accordingly
        if "```json" in result['answer'] and ("cols" in result['answer'] or "rows" in result['answer']):
            # This is likely a layout response, process it as such
            try:
                # Extract JSON content from the code block
                import json
                import re
                
                # Extract the JSON content from the markdown code block
                json_match = re.search(r'```json\s*\n(.*?)\n\s*```', result['answer'], re.DOTALL)
                if json_match:
                    json_content = json_match.group(1).strip()
                    parsed_json = json.loads(json_content)
                    
                    # Update Firebase with proper layout format
                    if db is not None and query_id and dashboard_context:
                        doc_ref = db.collection('rag_queries').document(query_id)
                        response_data = generate_layout_response(
                            doc_ref,
                            dashboard_context,
                            result['answer'],
                            parsed_json
                        )
                        
                        # Save the raw response to rag_responses collection
                        if db is not None:
                            # Save only the clean JSON object, not the markdown code block
                            save_response_to_firebase(query_text, json_content, "layout", query_id)
                        
                        return response_data
            except Exception as e:
                print(f"Error parsing layout JSON from RAG response: {str(e)}")
        
        # Update Firebase with the result (standard format if not layout or category)
        if db is not None and query_id:
            try:
                doc_ref = db.collection('rag_queries').document(query_id)
                doc_ref.update({
                    'answer': result,
                    'processed': True,
                    'processed_timestamp': firestore.SERVER_TIMESTAMP
                })
                
                # Save the raw response to rag_responses collection
                if db is not None:
                    save_response_to_firebase(query_text, json.dumps(result), "standard", query_id)
            except Exception as e:
                print(f"Error updating Firebase: {str(e)}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing Firebase query: {str(e)}")
        return jsonify({"error": str(e)}), 500

def save_response_to_firebase(query, raw_response, response_type, original_query_id=None):
    """Save raw RAG response to Firebase for future reference and analysis"""
    if db is None:
        print("Firebase not initialized, cannot save response")
        return
    
    try:
        # Create a new document in the rag_responses collection
        rag_responses_ref = db.collection('rag_responses')
        
        # Prepare the data to be saved
        response_data = {
            'query': query,
            'raw_response': raw_response,
            'response_type': response_type,
            'original_query_id': original_query_id,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'embeddings_count': len(collection.get()['ids']) if collection.get() and 'ids' in collection.get() else 0
        }
        
        # Add the document to the collection
        rag_responses_ref.add(response_data)
        print(f"Successfully saved response to rag_responses collection")
    
    except Exception as e:
        print(f"Error saving response to Firebase: {str(e)}")

@app.route('/api/save-recommendations', methods=['POST'])
def save_recommendations():
    """Save AI-generated recommendations to Firebase"""
    if db is None:
        print("Firebase not initialized, cannot save recommendations")
        return jsonify({"error": "Firebase not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Create a new document in the health_ai_recommendation collection
        recommendations_ref = db.collection('health_ai_recommendation')
        
        # Prepare the data to be saved
        recommendation_data = {
            'recommendations': data.get('recommendations', []),
            'bloodSugar': data.get('bloodSugar'),
            'exerciseMinutes': data.get('exerciseMinutes'),
            'mealType': data.get('mealType'),
            'medication': data.get('medication'),
            'status': data.get('status'),
            'query_id': data.get('query_id'),
            'health_context_id': data.get('health_context_id'),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': data.get('created_at') or firestore.SERVER_TIMESTAMP
        }
        
        # Add the document to the collection
        doc_ref = recommendations_ref.add(recommendation_data)
        print(f"Successfully saved recommendations to health_ai_recommendation collection with ID: {doc_ref[1].id}")
        
        return jsonify({
            "success": True,
            "message": "Recommendations saved successfully",
            "recommendation_id": doc_ref[1].id
        })
        
    except Exception as e:
        print(f"Error saving recommendations: {str(e)}")
        return jsonify({"error": f"Error saving recommendations: {str(e)}"}), 500

@app.route('/api/latest-work-context')
def latest_work_context():
    """API endpoint to fetch the latest work context from Firebase"""
    try:
        if db is None:
            return jsonify({"error": "Firebase not initialized"}), 500
            
        # Get the latest work context from Firebase
        work_ref = db.collection('work_context')
        tasks = work_ref.order_by('created_at', direction=firestore.Query.DESCENDING).limit(1).get()
        
        latest_task = None
        for doc in tasks:
            work_data = {
                'collaborators': doc.get('collaborators'),
                'created_at': doc.get('created_at'),
                'deadline': doc.get('deadline'),
                'notes': doc.get('notes'),
                'priority': doc.get('priority'),
                'status': doc.get('status'),
                'taskName': doc.get('taskName'),
                'timestamp': doc.get('timestamp'),
                'type': doc.get('type')
            }
            latest_task = work_data
            break
            
        return jsonify({"work_context": latest_task})
        
    except Exception as e:
        print(f"Error fetching work context: {str(e)}")
        return jsonify({"error": f"Error fetching work context: {str(e)}"}), 500

@app.route('/api/save-work-recommendations', methods=['POST'])
def save_work_recommendations():
    """Save AI-generated work recommendations to Firebase"""
    if db is None:
        print("Firebase not initialized, cannot save work recommendations")
        return jsonify({"error": "Firebase not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Create a new document in the work_ai_recommendation collection
        recommendations_ref = db.collection('work_ai_recommendation')
        
        # Prepare the data to be saved
        recommendation_data = {
            'recommendations': data.get('recommendations', []),
            'taskName': data.get('taskName'),
            'status': data.get('status'),
            'priority': data.get('priority'),
            'collaborators': data.get('collaborators'),
            'deadline': data.get('deadline'),
            'notes': data.get('notes'),
            'query_id': data.get('query_id'),
            'work_context_id': data.get('work_context_id'),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': data.get('created_at') or firestore.SERVER_TIMESTAMP
        }
        
        # Add the document to the collection
        doc_ref = recommendations_ref.add(recommendation_data)
        print(f"Successfully saved work recommendations to work_ai_recommendation collection with ID: {doc_ref[1].id}")
        
        return jsonify({
            "success": True,
            "message": "Work recommendations saved successfully",
            "recommendation_id": doc_ref[1].id
        })
        
    except Exception as e:
        print(f"Error saving work recommendations: {str(e)}")
        return jsonify({"error": f"Error saving work recommendations: {str(e)}"}), 500

@app.route('/api/latest-commute-context')
def latest_commute_context():
    """API endpoint to fetch the latest commute context from Firebase"""
    try:
        if db is None:
            return jsonify({"error": "Firebase not initialized"}), 500
            
        # Get the latest commute context from Firebase
        commute_ref = db.collection('commute_context')
        commutes = commute_ref.order_by('created_at', direction=firestore.Query.DESCENDING).limit(1).get()
        
        latest_commute = None
        for doc in commutes:
            commute_data = {
                'created_at': doc.get('created_at'),
                'duration': doc.get('duration'),
                'endLocation': doc.get('endLocation'),
                'notes': doc.get('notes'),
                'startLocation': doc.get('startLocation'),
                'timestamp': doc.get('timestamp'),
                'trafficCondition': doc.get('trafficCondition'),
                'transportMode': doc.get('transportMode'),
                'type': doc.get('type')
            }
            latest_commute = commute_data
            break
            
        return jsonify({"commute_context": latest_commute})
        
    except Exception as e:
        print(f"Error fetching commute context: {str(e)}")
        return jsonify({"error": f"Error fetching commute context: {str(e)}"}), 500

@app.route('/api/save-commute-recommendations', methods=['POST'])
def save_commute_recommendations():
    """Save AI-generated commute recommendations to Firebase"""
    if db is None:
        print("Firebase not initialized, cannot save commute recommendations")
        return jsonify({"error": "Firebase not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Create a new document in the commute_ai_recommendation collection
        recommendations_ref = db.collection('commute_ai_recommendation')
        
        # Prepare the data to be saved
        recommendation_data = {
            'recommendations': data.get('recommendations', []),
            'startLocation': data.get('startLocation'),
            'endLocation': data.get('endLocation'),
            'duration': data.get('duration'),
            'trafficCondition': data.get('trafficCondition'),
            'transportMode': data.get('transportMode'),
            'notes': data.get('notes'),
            'query_id': data.get('query_id'),
            'commute_context_id': data.get('commute_context_id'),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': data.get('created_at') or firestore.SERVER_TIMESTAMP
        }
        
        # Add the document to the collection
        doc_ref = recommendations_ref.add(recommendation_data)
        print(f"Successfully saved commute recommendations to commute_ai_recommendation collection with ID: {doc_ref[1].id}")
        
        return jsonify({
            "success": True,
            "message": "Commute recommendations saved successfully",
            "recommendation_id": doc_ref[1].id
        })
        
    except Exception as e:
        print(f"Error saving commute recommendations: {str(e)}")
        return jsonify({"error": f"Error saving commute recommendations: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 