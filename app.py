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
import sqlite3
import traceback

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

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence transformer model initialized")

# Initialize ChromaDB with logging
try:
    print("Initializing ChromaDB...")
    client = chromadb.Client(Settings(
        persist_directory="chroma_db",
        anonymized_telemetry=False
    ))
    print("ChromaDB client created successfully")

    # Create or get the collection with logging
    collection = client.get_or_create_collection(
        name="text_embeddings",
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection initialized. Current count: {len(collection.get()['ids'])}")

    # Add test data if collection is empty
    if len(collection.get()['ids']) == 0:
        print("Adding test data to empty collection...")
        test_data = [
            "This is a test document about health context.",
            "This is a test document about work context.",
            "This is a test document about commute context."
        ]
        embeddings = [model.encode(text).tolist() for text in test_data]
        collection.add(
            embeddings=embeddings,
            documents=test_data,
            ids=[f"test_{i+1}" for i in range(len(test_data))],
            metadatas=[
                {"source": "Test", "type": "health"},
                {"source": "Test", "type": "work"},
                {"source": "Test", "type": "commute"}
            ]
        )
        print(f"Added {len(test_data)} test documents")
except Exception as e:
    print(f"Error initializing ChromaDB: {str(e)}")
    raise e

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
        n_results=10,  # Increased to get more relevant contexts
        include=['embeddings', 'documents', 'metadatas', 'distances']
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
    
    # Process matching contexts with relevance scores
    matching_contexts = []
    for i in range(len(results['documents'][0])):
        # Convert distance to similarity score (1 - normalized_distance)
        similarity = 1 - min(results['distances'][0][i], 1.0)  # Ensure distance is not > 1
        context_type = results['metadatas'][0][i].get('type', 'unknown')
        
        matching_contexts.append({
            'type': context_type,
            'text': results['documents'][0][i],
            'relevance': similarity,
            'metadata': results['metadatas'][0][i]
        })
    
    # Sort contexts by relevance
    matching_contexts.sort(key=lambda x: x['relevance'], reverse=True)
    
    # Generate answer using Gemini with all relevant contexts
    context_texts = [ctx['text'] for ctx in matching_contexts]
    answer = generate_answer(query, context_texts)
    
    return {
        "answer": answer,
        "query_embedding_visualization": visualization_data["query_point"],
        "matching_contexts": matching_contexts
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

def generate_answer(query, context_texts):
    """Generate an answer using Gemini with multiple contexts."""
    if not context_texts:
        return "I couldn't find any relevant information to answer your question."
    
    # Create a prompt that includes all relevant contexts
    prompt = f"""You are a helpful AI assistant that answers questions based on the provided contexts. 
Your answers should be:
1. Accurate and based only on the provided contexts
2. Clear and well-structured
3. Include relevant quotes or references from the source contexts when appropriate

Contexts:
{chr(10).join([f"Context {i+1}:{chr(10)}{text}" for i, text in enumerate(context_texts)])}

Question: {query}

Please provide a comprehensive answer based on the above contexts. If the contexts don't contain enough information to answer the question fully, please state that explicitly."""

    response = model_gemini.generate_content(prompt)
    return response.text if response.text else "I couldn't generate a response based on the contexts."

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
        print("Fetching data from ChromaDB...")
        # Get all data from ChromaDB with embeddings included
        data = collection.get(include=['embeddings', 'documents', 'metadatas'])
        print(f"Retrieved {len(data['ids'] if 'ids' in data else [])} documents from ChromaDB")
        
        # If there's no data, return an empty list
        if not data or 'ids' not in data:
            print("No data found in ChromaDB")
            return jsonify([])
        
        # Format the data for the table
        formatted_data = []
        for i in range(len(data['ids'])):
            formatted_data.append({
                'id': data['ids'][i],
                'document': data['documents'][i][:100] + '...' if len(data['documents'][i]) > 100 else data['documents'][i],
                'embedding_size': len(data['embeddings'][i]),
                'source': data.get('metadatas', [{}])[i].get('source', 'Manual Input') if data.get('metadatas') else 'Manual Input'
            })
        
        print(f"Formatted {len(formatted_data)} documents for display")
        return jsonify(formatted_data)
    except Exception as e:
        print(f"Error in get_data: {str(e)}")
        return jsonify([]), 500

@app.route('/api/embeddings-visualization')
def get_embeddings_visualization():
    """Get embeddings visualization data with improved 3D visualization"""
    try:
        # Get all data from ChromaDB with embeddings included
        data = collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        # Get recommendations from Firebase
        recommendations = []
        if db is not None:
            try:
                # Get health recommendations
                health_recs = db.collection('health_ai_recommendation').order_by('created_at', direction=firestore.Query.DESCENDING).limit(10).get()
                for rec in health_recs:
                    rec_data = rec.to_dict()
                    text = f"Health Recommendation: Blood Sugar {rec_data.get('bloodSugar')}, Exercise {rec_data.get('exerciseMinutes')} mins, Recommendations: {', '.join(rec_data.get('recommendations', []))}"
                    recommendations.append({
                        'text': text,
                        'type': 'health_recommendation',
                        'source': 'AI Recommendation',
                        'id': rec.id
                    })

                # Get work recommendations
                work_recs = db.collection('work_ai_recommendation').order_by('created_at', direction=firestore.Query.DESCENDING).limit(10).get()
                for rec in work_recs:
                    rec_data = rec.to_dict()
                    text = f"Work Recommendation: Task {rec_data.get('taskName')}, Status {rec_data.get('status')}, Priority {rec_data.get('priority')}, Recommendations: {', '.join(rec_data.get('recommendations', []))}"
                    recommendations.append({
                        'text': text,
                        'type': 'work_recommendation',
                        'source': 'AI Recommendation',
                        'id': rec.id
                    })

                # Get commute recommendations
                commute_recs = db.collection('commute_ai_recommendation').order_by('created_at', direction=firestore.Query.DESCENDING).limit(10).get()
                for rec in commute_recs:
                    rec_data = rec.to_dict()
                    text = f"Commute Recommendation: From {rec_data.get('startLocation')} to {rec_data.get('endLocation')}, Mode {rec_data.get('transportMode')}, Recommendations: {', '.join(rec_data.get('recommendations', []))}"
                    recommendations.append({
                        'text': text,
                        'type': 'commute_recommendation',
                        'source': 'AI Recommendation',
                        'id': rec.id
                    })

            except Exception as e:
                print(f"Error fetching recommendations: {str(e)}")

        # If there's no data or no embeddings, return empty visualization
        if not data or 'embeddings' not in data or not data['embeddings']:
            return jsonify({
                "points": [],
                "variance_explained": [0, 0, 0],
                "metadata": {
                    "total_points": 0,
                    "categories": {},
                    "sources": {}
                }
            })

        # Add recommendation embeddings
        all_embeddings = data['embeddings']
        all_documents = data['documents']
        all_metadatas = data['metadatas']

        # Generate embeddings for recommendations
        for rec in recommendations:
            try:
                rec_embedding = model.encode(rec['text']).tolist()
                all_embeddings.append(rec_embedding)
                all_documents.append(rec['text'])
                all_metadatas.append({
                    'type': rec['type'],
                    'source': rec['source'],
                    'id': rec['id']
                })
            except Exception as e:
                print(f"Error encoding recommendation: {str(e)}")

        # Convert embeddings to numpy array
        embeddings = np.array(all_embeddings)
        
        # Determine number of components based on data size
        n_samples = embeddings.shape[0]
        n_features = embeddings.shape[1]
        n_components = min(3, n_samples - 1, n_features)
        
        if n_components < 1:
            return jsonify({
                "points": [],
                "variance_explained": [0, 0, 0],
                "metadata": {
                    "total_points": 0,
                    "categories": {},
                    "sources": {}
                }
            })
        
        # Apply PCA to reduce dimensions
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Create array for 3D coordinates
        coords_3d = np.zeros((reduced_embeddings.shape[0], 3))
        coords_3d[:, :n_components] = reduced_embeddings
        
        # Normalize coordinates to [-1, 1] range for better visualization
        for i in range(3):
            if np.max(np.abs(coords_3d[:, i])) > 0:
                coords_3d[:, i] = coords_3d[:, i] / np.max(np.abs(coords_3d[:, i]))
        
        # Prepare the visualization data with enhanced metadata
        points = []
        categories = {}
        sources = {}
        
        for i in range(len(coords_3d)):
            # Get metadata for this point
            metadata = all_metadatas[i]
            doc_type = metadata.get('type', 'unknown')
            source = metadata.get('source', 'unknown')
            
            # Update category and source counts
            categories[doc_type] = categories.get(doc_type, 0) + 1
            sources[source] = sources.get(source, 0) + 1
            
            # Create point data with enhanced information
            point_data = {
                'x': float(coords_3d[i, 0]),
                'y': float(coords_3d[i, 1]),
                'z': float(coords_3d[i, 2]),
                'text': all_documents[i][:200] + '...' if len(all_documents[i]) > 200 else all_documents[i],
                'source': source,
                'type': doc_type,
                'id': metadata.get('id', f'doc_{i}'),
                'size': 5,  # Default size
                'color': get_point_color(doc_type)  # Get color based on document type
            }
            points.append(point_data)
        
        # Prepare variance explained (pad with zeros if needed)
        variance_explained = list(pca.explained_variance_ratio_)
        while len(variance_explained) < 3:
            variance_explained.append(0.0)
        
        # Calculate additional metadata
        metadata = {
            "total_points": len(points),
            "categories": categories,
            "sources": sources,
            "dimensions": {
                "original": n_features,
                "reduced": n_components,
                "variance_explained": variance_explained
            }
        }
        
        return jsonify({
            "points": points,
            "variance_explained": variance_explained,
            "metadata": metadata
        })
        
    except Exception as e:
        print(f"Error in get_embeddings_visualization: {str(e)}")
        return jsonify({
            "error": str(e),
            "points": [],
            "variance_explained": [0, 0, 0],
            "metadata": {
                "total_points": 0,
                "categories": {},
                "sources": {}
            }
        }), 500

def get_point_color(doc_type):
    """Get color for different document types"""
    color_map = {
        'query': '#FF0000',  # Red for queries
        'health': '#00FF00',  # Green for health context
        'work': '#0000FF',   # Blue for work context
        'commute': '#FFA500', # Orange for commute context
        'layout': '#800080',  # Purple for layout data
        'health_recommendation': '#90EE90',  # Light green for health recommendations
        'work_recommendation': '#87CEEB',    # Light blue for work recommendations
        'commute_recommendation': '#FFB6C1',  # Light pink for commute recommendations
        'unknown': '#808080'  # Gray for unknown types
    }
    return color_map.get(doc_type, color_map['unknown'])

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
    """API endpoint to fetch the latest query from Firebase and local storage"""
    try:
        # First try to get from local storage
        local_data = get_from_local_db('query')
        if local_data:
            return jsonify({"query": local_data})
        
        # If not in local storage, get from Firebase
        if db is None:
            return jsonify({"error": "Firebase not initialized"}), 500
        
        queries_ref = db.collection('rag_queries')
        query = queries_ref.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1).get()
        
        latest_query = None
        for doc in query:
            query_data = doc.to_dict()
            if 'timestamp' in query_data and query_data['timestamp']:
                query_data['timestamp'] = query_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            query_data['id'] = doc.id
            latest_query = query_data
            break
        
        # Save to local storage
        if latest_query:
            save_to_local_db('query', latest_query)
            
            # Add to embeddings
            add_to_embeddings(
                latest_query['query'],
                {"source": "Firebase Query", "type": "query", "id": latest_query['id']}
            )
        
        return jsonify({"query": latest_query})
    
    except Exception as e:
        print(f"Error fetching from Firebase: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/latest-health-context')
def get_latest_health_context():
    try:
        # Get all health contexts from Firebase
        health_ref = db.collection('health_context').order_by('created_at', direction=firestore.Query.DESCENDING)
        docs = health_ref.get()
        
        contexts = []
        for doc in docs:
            context_data = doc.to_dict()
            context_data['id'] = doc.id
            
            # Save to local DB
            save_to_local_db('health_context', context_data)
            
            # Add to embeddings only if new
            context_text = f"Health Context: Blood Sugar {context_data.get('bloodSugar')}, Exercise {context_data.get('exerciseMinutes')} mins, Meal: {context_data.get('mealType')}, Medication: {context_data.get('medication')}, Notes: {context_data.get('notes', '')}"
            add_to_embeddings(context_text, {
                "id": doc.id,
                "type": "Health Context",
                "source": "Firebase"
            })
            
            contexts.append(context_data)
            
        return jsonify({"health_contexts": contexts})
            
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/latest-work-context')
def get_latest_work_context():
    try:
        # Get all work contexts from Firebase
        work_ref = db.collection('work_context').order_by('created_at', direction=firestore.Query.DESCENDING)
        docs = work_ref.get()
        
        contexts = []
        for doc in docs:
            context_data = doc.to_dict()
            context_data['id'] = doc.id
            
            # Save to local DB
            save_to_local_db('work_context', context_data)
            
            # Add to embeddings only if new
            context_text = f"Work Context: Task {context_data.get('taskName')}, Status {context_data.get('status')}, Priority {context_data.get('priority')}, Deadline {context_data.get('deadline')}, Team: {context_data.get('collaborators')}"
            add_to_embeddings(context_text, {
                "id": doc.id,
                "type": "Work Context",
                "source": "Firebase"
            })
            
            contexts.append(context_data)
            
        return jsonify({"work_contexts": contexts})
            
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/latest-commute-context')
def get_latest_commute_context():
    try:
        # Get all commute contexts from Firebase
        commute_ref = db.collection('commute_context').order_by('created_at', direction=firestore.Query.DESCENDING)
        docs = commute_ref.get()
        
        contexts = []
        for doc in docs:
            context_data = doc.to_dict()
            context_data['id'] = doc.id
            
            # Save to local DB
            save_to_local_db('commute_context', context_data)
            
            # Add to embeddings only if new
            context_text = f"Commute Context: From {context_data.get('startLocation')} to {context_data.get('endLocation')}, Mode: {context_data.get('transportMode')}, Duration: {context_data.get('duration')} mins, Traffic: {context_data.get('trafficCondition')}, Notes: {context_data.get('notes', '')}"
            add_to_embeddings(context_text, {
                "id": doc.id,
                "type": "Commute Context",
                "source": "Firebase"
            })
            
            contexts.append(context_data)
            
        return jsonify({"commute_contexts": contexts})
            
    except Exception as e:
        return jsonify({"error": str(e)})

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
        
        # Process the query using standard RAG
        result = process_search_query(query_text)
        
        # Save the result to Firebase if available
        query_id = request.form.get('query_id')
        if db is not None and query_id:
            try:
                # Save the query result
                doc_ref = db.collection('rag_queries').document(query_id)
                doc_ref.set({
                    'query': query_text,
                    'answer': result['answer'],
                    'processed': True,
                    'processed_timestamp': firestore.SERVER_TIMESTAMP,
                    'matching_contexts': result.get('matching_contexts', [])
                }, merge=True)
                
                # Save individual context responses
                health_context_id = request.form.get('health_context_id')
                work_context_id = request.form.get('work_context_id')
                commute_context_id = request.form.get('commute_context_id')
                
                # Filter matching contexts by type
                health_contexts = [ctx for ctx in result.get('matching_contexts', []) if ctx['type'] == 'health']
                work_contexts = [ctx for ctx in result.get('matching_contexts', []) if ctx['type'] == 'work']
                commute_contexts = [ctx for ctx in result.get('matching_contexts', []) if ctx['type'] == 'commute']
                
                # Save health context response
                if health_context_id and health_contexts:
                    db.collection('health_responses').add({
                        'query_id': query_id,
                        'health_context_id': health_context_id,
                        'contexts': health_contexts,
                        'created_at': firestore.SERVER_TIMESTAMP
                    })
                
                # Save work context response
                if work_context_id and work_contexts:
                    db.collection('work_responses').add({
                        'query_id': query_id,
                        'work_context_id': work_context_id,
                        'contexts': work_contexts,
                        'created_at': firestore.SERVER_TIMESTAMP
                    })
                
                # Save commute context response
                if commute_context_id and commute_contexts:
                    db.collection('commute_responses').add({
                        'query_id': query_id,
                        'commute_context_id': commute_context_id,
                        'contexts': commute_contexts,
                        'created_at': firestore.SERVER_TIMESTAMP
                    })
                
                # Add query_id to the result
                result['query_id'] = query_id
                
            except Exception as e:
                print(f"Error saving to Firebase: {str(e)}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
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

def sync_local_recommendations_with_firebase():
    """Sync local work recommendations with Firebase"""
    try:
        # Get local recommendations from SQLite
        conn = sqlite3.connect('local_data.db')
        c = conn.cursor()
        c.execute('SELECT * FROM work_context ORDER BY created_at DESC')
        local_recommendations = c.fetchall()
        conn.close()

        if not local_recommendations:
            print("No local recommendations to sync")
            return

        # Get Firebase recommendations
        recommendations_ref = db.collection('work_ai_recommendation')
        
        for local_rec in local_recommendations:
            # Convert SQLite row to dict
            columns = ['id', 'task_name', 'status', 'priority', 'collaborators', 
                      'deadline', 'notes', 'timestamp', 'type', 'created_at']
            rec_dict = dict(zip(columns, local_rec))
            
            # Check if recommendation exists in Firebase
            query = recommendations_ref.where('work_context_id', '==', rec_dict['id']).limit(1).get()
            
            recommendation_data = {
                'taskName': rec_dict['task_name'],
                'status': rec_dict['status'],
                'priority': rec_dict['priority'],
                'collaborators': rec_dict['collaborators'],
                'deadline': rec_dict['deadline'],
                'notes': rec_dict['notes'],
                'work_context_id': rec_dict['id'],
                'timestamp': firestore.SERVER_TIMESTAMP,
                'created_at': firestore.SERVER_TIMESTAMP
            }
            
            if not query:  # Document doesn't exist in Firebase
                print(f"Adding local recommendation {rec_dict['id']} to Firebase")
                doc_ref = recommendations_ref.add(recommendation_data)
                print(f"Created new recommendation in Firebase with ID: {doc_ref[1].id}")
            else:
                # Update existing document
                for doc in query:
                    print(f"Updating existing recommendation in Firebase with ID: {doc.id}")
                    doc.reference.update(recommendation_data)
                    
    except Exception as e:
        print(f"Error syncing recommendations with Firebase: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")

@app.route('/api/save-work-recommendations', methods=['POST'])
def save_work_recommendations():
    """Save AI-generated work recommendations to Firebase"""
    print("Received request to save work recommendations")
    
    if db is None:
        print("Error: Firebase DB is not initialized")
        return jsonify({"error": "Firebase not initialized"}), 500
    
    try:
        data = request.get_json()
        print(f"Received work recommendation data: {json.dumps(data, indent=2)}")
        
        if not data:
            print("Error: No data provided in request")
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
            'created_at': firestore.SERVER_TIMESTAMP  # Force server timestamp
        }
        
        # First, try to update existing document if work_context_id exists
        if data.get('work_context_id'):
            # Query for existing recommendation with same work_context_id
            existing_docs = recommendations_ref.where('work_context_id', '==', data['work_context_id']).limit(1).get()
            
            for doc in existing_docs:
                # Update existing document
                doc.reference.update(recommendation_data)
                print(f"Updated existing recommendation document with ID: {doc.id}")
                return jsonify({
                    "success": True,
                    "message": "Work recommendations updated successfully",
                    "recommendation_id": doc.id
                })
        
        # If no existing document found or no work_context_id, create new document
        doc_ref = recommendations_ref.add(recommendation_data)
        print(f"Created new recommendation document with ID: {doc_ref[1].id}")
        
        # Verify the document was saved
        saved_doc = doc_ref[1].get()
        if not saved_doc.exists:
            raise Exception("Document was not saved to Firebase")
            
        print(f"Verified document exists in Firebase with data: {saved_doc.to_dict()}")
        
        # Sync all local recommendations with Firebase
        sync_local_recommendations_with_firebase()
        
        return jsonify({
            "success": True,
            "message": "Work recommendations saved successfully",
            "recommendation_id": doc_ref[1].id
        })
    except Exception as e:
        error_msg = f"Error saving work recommendations: {str(e)}"
        print(f"Error details: {error_msg}")
        print(f"Stack trace: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

@app.route('/api/save-commute-recommendations', methods=['POST'])
def save_commute_recommendations():
    """Save AI-generated commute recommendations to Firebase"""
    if db is None:
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

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('local_data.db')
    c = conn.cursor()
    
    # Create tables for different types of data
    c.execute('''CREATE TABLE IF NOT EXISTS queries
                 (id TEXT PRIMARY KEY, query TEXT, timestamp TEXT, 
                  source TEXT, processed BOOLEAN, answer TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS health_context
                 (id TEXT PRIMARY KEY, blood_sugar TEXT, created_at TEXT,
                  exercise_minutes TEXT, meal_type TEXT, medication TEXT,
                  notes TEXT, timestamp TEXT, type TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS work_context
                 (id TEXT PRIMARY KEY, task_name TEXT, status TEXT,
                  priority TEXT, collaborators TEXT, deadline TEXT,
                  notes TEXT, timestamp TEXT, type TEXT, created_at TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS commute_context
                 (id TEXT PRIMARY KEY, duration TEXT, end_location TEXT,
                  notes TEXT, start_location TEXT, timestamp TEXT,
                  traffic_condition TEXT, transport_mode TEXT, type TEXT,
                  created_at TEXT)''')
    
    conn.commit()
    conn.close()

# Initialize the database when the app starts
init_db()

def save_to_local_db(data_type, data):
    """Save data to local SQLite database"""
    conn = sqlite3.connect('local_data.db')
    c = conn.cursor()
    
    try:
        # Convert Firebase timestamps to strings
        if isinstance(data.get('timestamp'), datetime.datetime):
            data['timestamp'] = data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        if isinstance(data.get('created_at'), datetime.datetime):
            data['created_at'] = data['created_at'].strftime('%Y-%m-%d %H:%M:%S')
            
        if data_type == 'query':
            c.execute('''INSERT OR REPLACE INTO queries 
                        (id, query, timestamp, source, processed, answer)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                     (str(data.get('id')), str(data.get('query')), str(data.get('timestamp')),
                      str(data.get('source')), bool(data.get('processed', False)),
                      json.dumps(data.get('answer', {}))))
        
        elif data_type == 'health_context':
            c.execute('''INSERT OR REPLACE INTO health_context 
                        (id, blood_sugar, created_at, exercise_minutes,
                         meal_type, medication, notes, timestamp, type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (str(data.get('id')), str(data.get('bloodSugar')), str(data.get('created_at')),
                      str(data.get('exerciseMinutes')), str(data.get('mealType')),
                      str(data.get('medication')), str(data.get('notes', '')), str(data.get('timestamp')),
                      str(data.get('type'))))
        
        elif data_type == 'work_context':
            c.execute('''INSERT OR REPLACE INTO work_context 
                        (id, task_name, status, priority, collaborators,
                         deadline, notes, timestamp, type, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (str(data.get('id')), str(data.get('taskName')), str(data.get('status')),
                      str(data.get('priority')), str(data.get('collaborators')),
                      str(data.get('deadline')), str(data.get('notes', '')), str(data.get('timestamp')),
                      str(data.get('type')), str(data.get('created_at'))))
        
        elif data_type == 'commute_context':
            c.execute('''INSERT OR REPLACE INTO commute_context 
                        (id, duration, end_location, notes, start_location,
                         timestamp, traffic_condition, transport_mode, type, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                     (str(data.get('id')), str(data.get('duration')), str(data.get('endLocation')),
                      str(data.get('notes', '')), str(data.get('startLocation')), str(data.get('timestamp')),
                      str(data.get('trafficCondition')), str(data.get('transportMode')),
                      str(data.get('type')), str(data.get('created_at'))))
        
        conn.commit()
    except Exception as e:
        print(f"Error saving to local database: {str(e)}")
    finally:
        conn.close()

def get_from_local_db(data_type, limit=1):
    """Retrieve data from local SQLite database"""
    conn = sqlite3.connect('local_data.db')
    c = conn.cursor()
    
    try:
        if data_type == 'query':
            c.execute('SELECT * FROM queries ORDER BY timestamp DESC LIMIT ?', (limit,))
        elif data_type == 'health_context':
            c.execute('SELECT * FROM health_context ORDER BY created_at DESC LIMIT ?', (limit,))
        elif data_type == 'work_context':
            c.execute('SELECT * FROM work_context ORDER BY created_at DESC LIMIT ?', (limit,))
        elif data_type == 'commute_context':
            c.execute('SELECT * FROM commute_context ORDER BY created_at DESC LIMIT ?', (limit,))
        
        rows = c.fetchall()
        if not rows:
            return None
        
        # Convert row to dictionary
        columns = [description[0] for description in c.description]
        data = dict(zip(columns, rows[0]))
        
        # Convert JSON strings back to objects
        if data_type == 'query' and data.get('answer'):
            data['answer'] = json.loads(data['answer'])
        
        return data
    except Exception as e:
        print(f"Error retrieving from local database: {str(e)}")
        return None
    finally:
        conn.close()

def check_existing_embedding(metadata):
    """Check if an embedding with the same metadata already exists"""
    try:
        # Get all documents and check metadata manually since ChromaDB where clause is limited
        results = collection.get(include=["metadatas"])
        if not results or 'metadatas' not in results:
            return False
            
        # Check each metadata for matching id and type
        for meta in results['metadatas']:
            if (meta.get('id') == metadata.get('id') and 
                meta.get('type') == metadata.get('type')):
                return True
        return False
    except Exception as e:
        print(f"Error checking existing embedding: {str(e)}")
        return False

def add_to_embeddings(text, metadata):
    """Add text to ChromaDB embeddings if it doesn't already exist"""
    try:
        # Check if embedding already exists
        if check_existing_embedding(metadata):
            print(f"Embedding already exists for {metadata.get('type')} with ID {metadata.get('id')}")
            return False
            
        # Generate embedding
        embedding = model.encode(text).tolist()
        
        # Generate a unique document ID based on metadata
        doc_id = f"{metadata.get('type', 'doc')}_{metadata.get('id', 'unknown')}".replace(' ', '_').lower()
        
        # Add to ChromaDB
        collection.add(
            embeddings=[embedding],
            documents=[text],
            ids=[doc_id],
            metadatas=[metadata]
        )
        print(f"Added new embedding for {metadata.get('type')} with ID {metadata.get('id')}")
        return True
    except Exception as e:
        print(f"Error adding to embeddings: {str(e)}")
        return False

@app.route('/api/send-work-recommendation', methods=['POST'])
def send_work_recommendation():
    """Explicitly send a work recommendation to Firebase"""
    try:
        print("Received request to send work recommendation")
        data = request.get_json()
        print(f"Request data: {json.dumps(data, indent=2)}")
        
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Required fields
        required_fields = ['taskName', 'status', 'priority', 'recommendations']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Create recommendation data
        recommendation_data = {
            'taskName': data['taskName'],
            'status': data['status'],
            'priority': data['priority'],
            'recommendations': data['recommendations'],
            'collaborators': data.get('collaborators', ''),
            'deadline': data.get('deadline', ''),
            'notes': data.get('notes', ''),
            'work_context_id': data.get('work_context_id'),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': firestore.SERVER_TIMESTAMP
        }

        print(f"Preparing to save recommendation data: {json.dumps(recommendation_data, indent=2, default=str)}")

        # Add to Firebase
        recommendations_ref = db.collection('work_ai_recommendation')
        doc_ref = recommendations_ref.add(recommendation_data)
        doc_id = doc_ref[1].id
        print(f"Created new recommendation document with ID: {doc_id}")
        
        # Verify the save
        saved_doc = doc_ref[1].get()
        if not saved_doc.exists:
            raise Exception("Failed to save recommendation to Firebase")

        saved_data = saved_doc.to_dict()
        print(f"Verified saved document data: {json.dumps(saved_data, indent=2, default=str)}")

        return jsonify({
            "success": True,
            "message": "Recommendation sent to Firebase successfully",
            "recommendation_id": doc_id,
            "saved_data": saved_data
        })

    except Exception as e:
        error_msg = f"Error sending recommendation to Firebase: {str(e)}"
        print(f"Error details: {error_msg}")
        print(f"Stack trace: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

@app.route('/api/send-health-recommendation', methods=['POST'])
def send_health_recommendation():
    """Explicitly send a health recommendation to Firebase"""
    try:
        print("Received request to send health recommendation")
        data = request.get_json()
        print(f"Request data: {json.dumps(data, indent=2)}")
        
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Required fields
        required_fields = ['bloodSugar', 'exerciseMinutes', 'mealType', 'recommendations']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Create recommendation data
        recommendation_data = {
            'bloodSugar': data['bloodSugar'],
            'exerciseMinutes': data['exerciseMinutes'],
            'mealType': data['mealType'],
            'recommendations': data['recommendations'],
            'medication': data.get('medication', ''),
            'notes': data.get('notes', ''),
            'health_context_id': data.get('health_context_id'),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': firestore.SERVER_TIMESTAMP
        }

        print(f"Preparing to save health recommendation data: {json.dumps(recommendation_data, indent=2, default=str)}")

        # Add to Firebase
        recommendations_ref = db.collection('health_ai_recommendation')
        doc_ref = recommendations_ref.add(recommendation_data)
        doc_id = doc_ref[1].id
        print(f"Created new health recommendation document with ID: {doc_id}")
        
        # Verify the save
        saved_doc = doc_ref[1].get()
        if not saved_doc.exists:
            raise Exception("Failed to save health recommendation to Firebase")

        saved_data = saved_doc.to_dict()
        print(f"Verified saved document data: {json.dumps(saved_data, indent=2, default=str)}")

        return jsonify({
            "success": True,
            "message": "Health recommendation sent to Firebase successfully",
            "recommendation_id": doc_id,
            "saved_data": saved_data
        })

    except Exception as e:
        error_msg = f"Error sending health recommendation to Firebase: {str(e)}"
        print(f"Error details: {error_msg}")
        print(f"Stack trace: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

@app.route('/api/send-commute-recommendation', methods=['POST'])
def send_commute_recommendation():
    """Explicitly send a commute recommendation to Firebase"""
    try:
        print("Received request to send commute recommendation")
        data = request.get_json()
        print(f"Request data: {json.dumps(data, indent=2)}")
        
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Required fields
        required_fields = ['startLocation', 'endLocation', 'duration', 'recommendations']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Create recommendation data
        recommendation_data = {
            'startLocation': data['startLocation'],
            'endLocation': data['endLocation'],
            'duration': data['duration'],
            'recommendations': data['recommendations'],
            'trafficCondition': data.get('trafficCondition', ''),
            'transportMode': data.get('transportMode', ''),
            'notes': data.get('notes', ''),
            'commute_context_id': data.get('commute_context_id'),
            'timestamp': firestore.SERVER_TIMESTAMP,
            'created_at': firestore.SERVER_TIMESTAMP
        }

        print(f"Preparing to save commute recommendation data: {json.dumps(recommendation_data, indent=2, default=str)}")

        # Add to Firebase
        recommendations_ref = db.collection('commute_ai_recommendation')
        doc_ref = recommendations_ref.add(recommendation_data)
        doc_id = doc_ref[1].id
        print(f"Created new commute recommendation document with ID: {doc_id}")
        
        # Verify the save
        saved_doc = doc_ref[1].get()
        if not saved_doc.exists:
            raise Exception("Failed to save commute recommendation to Firebase")

        saved_data = saved_doc.to_dict()
        print(f"Verified saved document data: {json.dumps(saved_data, indent=2, default=str)}")

        return jsonify({
            "success": True,
            "message": "Commute recommendation sent to Firebase successfully",
            "recommendation_id": doc_id,
            "saved_data": saved_data
        })

    except Exception as e:
        error_msg = f"Error sending commute recommendation to Firebase: {str(e)}"
        print(f"Error details: {error_msg}")
        print(f"Stack trace: {traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 